from __future__ import annotations

import traceback

from agent.config import load_config
from agent.models import AgentRunSummary, ProbeResult
from agent.probes import ProbeExecutor
from agent.spec import load_target_spec
from agent.state import load_state, save_state
from agent.utils import atomic_write_json, monotonic_seconds, utc_now_iso
from llm.reasoning import maybe_generate_reasoning


def _serialize_probe_result(result: ProbeResult) -> dict:
    return {
        "status": result.status,
        "value": result.value,
        "unit": result.unit,
        "method": result.method,
        "confidence": result.confidence,
        "trials": result.trials,
        "summary": result.summary,
        "evidence": result.evidence,
        "error": result.error,
    }


def _build_run_summary(
    *,
    student_id: str,
    started_at: str,
    finished_at: str,
    duration_seconds: float,
    results: list[ProbeResult],
) -> AgentRunSummary:
    success_count = sum(1 for item in results if item.status == "ok")
    failure_count = sum(1 for item in results if item.status != "ok")
    return AgentRunSummary(
        student_id=student_id,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        target_count=len(results),
        success_count=success_count,
        failure_count=failure_count,
    )


def run_agent() -> int:
    config = load_config()
    state = load_state(config.state_path)

    started_at = utc_now_iso()
    start_monotonic = monotonic_seconds()

    try:
        spec = load_target_spec(config.target_spec_path)
        probe_executor = ProbeExecutor(config)

        probe_results: list[ProbeResult] = []
        for target in spec.targets:
            probe_results.append(probe_executor.run_target(target))

        finished_at = utc_now_iso()
        duration_seconds = monotonic_seconds() - start_monotonic

        summary = _build_run_summary(
            student_id=config.student_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            results=probe_results,
        )

        numeric_results = {
            item.target: item.value if item.status == "ok" else None
            for item in probe_results
        }
        detailed_results = {
            item.target: _serialize_probe_result(item)
            for item in probe_results
        }

        base_payload = {
            "student_id": config.student_id,
            "target_spec_path": str(config.target_spec_path),
            "results": numeric_results,
            "details": detailed_results,
            "summary": {
                "started_at": summary.started_at,
                "finished_at": summary.finished_at,
                "duration_seconds": summary.duration_seconds,
                "target_count": summary.target_count,
                "success_count": summary.success_count,
                "failure_count": summary.failure_count,
            },
        }

        reasoning = maybe_generate_reasoning(base_payload)
        output_payload = dict(base_payload)
        output_payload["reasoning"] = reasoning

        atomic_write_json(config.output_path, output_payload)

        state["last_error"] = None
        state.setdefault("runs", []).append(
            {
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": duration_seconds,
                "target_spec_path": str(config.target_spec_path),
                "output_path": str(config.output_path),
                "success_count": summary.success_count,
                "failure_count": summary.failure_count,
            }
        )
        save_state(config.state_path, state)

        print(f"Agent completed. Output written to: {config.output_path}")
        print(f"Targets processed: {summary.target_count}, successful: {summary.success_count}, failed: {summary.failure_count}")
        return 0

    except Exception as exc:
        finished_at = utc_now_iso()
        duration_seconds = monotonic_seconds() - start_monotonic

        state["last_error"] = {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_state(config.state_path, state)

        failure_payload = {
            "student_id": config.student_id,
            "results": {},
            "details": {},
            "summary": {
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": duration_seconds,
                "target_count": 0,
                "success_count": 0,
                "failure_count": 0,
            },
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(config.output_path, failure_payload)

        print(f"Agent failed. Output written to: {config.output_path}")
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(run_agent())
