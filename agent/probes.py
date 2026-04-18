from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.config import AgentConfig
from agent.models import ProbeResult, TargetPlan
from agent.planner import TargetPlanner
from agent.utils import estimate_confidence, extract_last_json_object, summarize_trials
from llm.openai_client import get_openai_client
from runner.run import (
    aggregate_metric_records,
    compile_cuda_source,
    parse_ncu_csv,
    profile_with_ncu,
    run_binary,
)


@dataclass
class PreparedBinary:
    source_path: Path
    binary_path: Path
    source_hash: str


class ProbeExecutor:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.source_dir = config.generated_dir / "sources"
        self.binary_dir = config.generated_dir / "bin"
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.binary_dir.mkdir(parents=True, exist_ok=True)
        self._compiled: dict[str, PreparedBinary] = {}
        self._builtin_handlers = self._handlers()
        self._planner = TargetPlanner(config, list(self._builtin_handlers.keys()))
        self._llm_client = get_openai_client()
        self._llm_model = os.getenv("BASE_MODEL", "").strip()
        llm_flag = os.getenv("LLM_BENCHMARK_ENABLED", "1").strip().lower()
        self._llm_benchmark_enabled = llm_flag not in {"0", "false", "no"}
        self._llm_benchmark_enabled = self._llm_benchmark_enabled and self._llm_client is not None and bool(self._llm_model)

    def run_target(self, target: str) -> ProbeResult:
        plan = self._planner.plan_target(target)

        try:
            result = self._execute_plan(plan)
            result.evidence.setdefault("plan", plan.as_dict())
            return result
        except Exception as exc:
            return ProbeResult(
                target=target,
                status="failed",
                value=None,
                unit="",
                method=plan.kind,
                confidence=0.0,
                evidence={"plan": plan.as_dict()},
                error=str(exc),
            )

    def _execute_plan(self, plan: TargetPlan) -> ProbeResult:
        if plan.kind == "builtin_probe":
            probe_name = (plan.probe_name or "").strip().lower()
            handler = self._builtin_handlers.get(probe_name)
            if handler is None:
                raise RuntimeError(f"Unknown built-in probe: {probe_name}")
            return handler(plan.target)

        if plan.kind == "device_attribute":
            attribute_name = (plan.attribute_name or "").strip().lower()
            if not attribute_name:
                raise RuntimeError("Device attribute plan missing attribute_name")
            return self._probe_device_attribute(plan.target, attribute_name, plan.unit_hint)

        if plan.kind == "ncu_metric":
            metric_name = (plan.metric_name or "").strip()
            if not metric_name:
                raise RuntimeError("NCU metric plan missing metric_name")
            return self._probe_ncu_metric(plan.target, metric_name, plan.unit_hint)

        raise RuntimeError(f"Unsupported plan kind: {plan.kind}")

    def _handlers(self) -> dict[str, Callable[[str], ProbeResult]]:
        return {
            "physical_sm_count": self._probe_physical_sm_count,
            "actual_core_clock_mhz": self._probe_actual_core_clock_mhz,
            "actual_boost_clock_mhz": self._probe_actual_core_clock_mhz,
            "peak_fp32_tflops": self._probe_peak_fp32_tflops,
            "l1_latency_cycles": self._probe_l1_latency_cycles,
            "l2_latency_cycles": self._probe_l2_latency_cycles,
            "dram_latency_cycles": self._probe_dram_latency_cycles,
            "l2_cache_capacity_kb": self._probe_l2_cache_capacity_kb,
            "global_bandwidth_gbps": self._probe_global_bandwidth_gbps,
            "shared_bandwidth_gbps": self._probe_shared_bandwidth_gbps,
            "bank_conflict_penalty_cycles": self._probe_bank_conflict_penalty_cycles,
            "max_shmem_per_block_kb": self._probe_max_shmem_per_block_kb,
        }

    def _probe_device_attribute(self, original_target: str, attribute_name: str, unit_hint: str) -> ProbeResult:
        prepared = self._prepare_binary("probe_device_attribute", _source_device_attribute())
        values: list[float] = []
        evidences: list[dict[str, Any]] = []

        for _ in range(max(1, self.config.max_trials)):
            payload, evidence = self._run_binary_json(prepared, args=[attribute_name])
            if "value" not in payload:
                raise RuntimeError(f"Attribute probe output missing 'value': {payload}")
            values.append(float(payload["value"]))
            evidence["payload"] = payload
            evidences.append(evidence)

        summary = summarize_trials(values)
        payload_unit = ""
        if evidences and isinstance(evidences[-1].get("payload"), dict):
            payload_unit = str(evidences[-1]["payload"].get("unit", "")).strip()

        return self._build_result(
            target=original_target,
            value=summary["median"],
            unit=payload_unit or unit_hint,
            method=f"device-attribute:{attribute_name}",
            trials=values,
            evidence={"attribute_name": attribute_name, "runs": evidences},
        )

    def _probe_ncu_metric(self, original_target: str, metric_name: str, unit_hint: str) -> ProbeResult:
        if self._llm_benchmark_enabled:
            try:
                return self._probe_ncu_metric_llm_generated(original_target, metric_name, unit_hint)
            except Exception as exc:
                fallback = self._probe_ncu_metric_static(original_target, metric_name, unit_hint)
                fallback.evidence["llm_benchmark_error"] = str(exc)
                fallback.evidence["llm_benchmark_fallback"] = True
                return fallback

        return self._probe_ncu_metric_static(original_target, metric_name, unit_hint)

    def _probe_ncu_metric_static(self, original_target: str, metric_name: str, unit_hint: str) -> ProbeResult:
        prepared = self._prepare_binary("probe_metric_workload", _source_metric_workload())

        profile = profile_with_ncu(
            prepared.binary_path,
            args=["16777216", "30", "1024", "256"],
            metrics=[metric_name],
            timeout_s=self.config.profile_timeout_s,
        )

        records = parse_ncu_csv(profile.stdout)
        aggregate = aggregate_metric_records(records)

        matched_metric, stats = self._match_metric_stats(aggregate, metric_name)
        if stats is None:
            raise RuntimeError(
                f"Requested metric '{metric_name}' not found in ncu output. Available metrics: {sorted(aggregate.keys())}"
            )

        value = float(stats["median"])
        unit = str(stats.get("unit", "")).strip() or unit_hint

        return self._build_result(
            target=original_target,
            value=value,
            unit=unit,
            method=f"ncu-metric:{matched_metric}",
            trials=[value],
            evidence={
                "requested_metric": metric_name,
                "matched_metric": matched_metric,
                "profile_duration_seconds": profile.duration_seconds,
                "record_count": len(records),
                "aggregate": aggregate,
                "benchmark_source": "builtin_metric_workload",
            },
        )

    def _probe_ncu_metric_llm_generated(self, original_target: str, metric_name: str, unit_hint: str) -> ProbeResult:
        prepared, generation = self._prepare_llm_metric_binary(original_target, metric_name)

        profile = profile_with_ncu(
            prepared.binary_path,
            args=None,
            metrics=[metric_name],
            timeout_s=self.config.profile_timeout_s,
        )

        records = parse_ncu_csv(profile.stdout)
        aggregate = aggregate_metric_records(records)
        matched_metric, stats = self._match_metric_stats(aggregate, metric_name)
        if stats is None:
            raise RuntimeError(
                f"Requested metric '{metric_name}' not found in ncu output. Available metrics: {sorted(aggregate.keys())}"
            )

        value = float(stats["median"])
        unit = str(stats.get("unit", "")).strip() or unit_hint

        return self._build_result(
            target=original_target,
            value=value,
            unit=unit,
            method=f"ncu-metric-llm-benchmark:{matched_metric}",
            trials=[value],
            evidence={
                "requested_metric": metric_name,
                "matched_metric": matched_metric,
                "profile_duration_seconds": profile.duration_seconds,
                "record_count": len(records),
                "aggregate": aggregate,
                "benchmark_source": "llm_generated",
                "benchmark_path": str(prepared.source_path),
                "generation": generation,
            },
        )

    def _prepare_llm_metric_binary(self, target: str, metric_name: str) -> tuple[PreparedBinary, dict[str, Any]]:
        if self._llm_client is None or not self._llm_model:
            raise RuntimeError("LLM benchmark generation is disabled or OpenAI client is unavailable")

        last_error: str | None = None
        attempts: list[dict[str, Any]] = []

        for attempt in range(1, 4):
            source = self._generate_llm_benchmark_source(target=target, metric_name=metric_name, error_context=last_error)
            source_hash = self._hash_text(source)
            name = f"probe_metric_llm_{source_hash[:12]}"

            attempt_evidence: dict[str, Any] = {
                "attempt": attempt,
                "name": name,
                "source_hash": source_hash,
            }

            try:
                prepared = self._prepare_binary(name, source)
                attempt_evidence["compiled"] = True
                attempts.append(attempt_evidence)
                return prepared, {"attempts": attempts}
            except Exception as exc:
                attempt_evidence["compiled"] = False
                attempt_evidence["error"] = str(exc)
                attempts.append(attempt_evidence)
                last_error = str(exc)

        raise RuntimeError(f"Failed to compile LLM-generated benchmark for metric '{metric_name}'")

    def _generate_llm_benchmark_source(self, *, target: str, metric_name: str, error_context: str | None) -> str:
        if self._llm_client is None or not self._llm_model:
            raise RuntimeError("LLM benchmark generation requested but client/model is unavailable")

        prompt_path = self.config.prompt_dir / "generate_benchmark.txt"
        if prompt_path.exists():
            template = prompt_path.read_text(encoding="utf-8")
        else:
            template = (
                "Generate a complete CUDA benchmark source file (.cu).\\n"
                "Target label: {target}\\n"
                "Primary metric: {metric_name}\\n"
                "Requirements:\\n"
                "- Return raw CUDA source only (no markdown).\\n"
                "- Include at least one __global__ kernel and a main function.\\n"
                "- Compile with nvcc -O3 -std=c++17.\\n"
                "- No command-line arguments are required.\\n"
                "- Print one JSON line at the end: {{\"status\":\"ok\",\"elapsed_ms\":number}}.\\n"
                "- Do not use #include <cuda/wmma.hpp>.\\n"
                "Previous error to fix (if any):\\n{error_context}\\n"
            )

        prompt = template.format(
            target=target,
            metric_name=metric_name,
            error_context=error_context or "none",
        )

        response = self._llm_client.chat.completions.create(
            model=self._llm_model,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM returned empty benchmark source")

        source = self._extract_code_block(content)
        if "__global__" not in source or "main(" not in source:
            raise RuntimeError("Generated benchmark source is missing a CUDA kernel or main()")

        return source

    def _extract_code_block(self, text: str) -> str:
        stripped = text.strip()
        cuda_fence = "```cuda"
        if cuda_fence in stripped:
            start = stripped.find(cuda_fence) + len(cuda_fence)
            end = stripped.find("```", start)
            if end == -1:
                end = len(stripped)
            return stripped[start:end].strip()

        generic_fence = "```"
        if generic_fence in stripped:
            start = stripped.find(generic_fence) + len(generic_fence)
            end = stripped.find(generic_fence, start)
            if end == -1:
                end = len(stripped)
            return stripped[start:end].strip()

        return stripped

    def _match_metric_stats(
        self,
        aggregate: dict[str, dict[str, Any]],
        metric_name: str,
    ) -> tuple[str, dict[str, Any] | None]:
        stats = aggregate.get(metric_name)
        matched_metric = metric_name

        if stats is None:
            lowered = metric_name.lower()
            for key, value in aggregate.items():
                if key.lower() == lowered:
                    stats = value
                    matched_metric = key
                    break

        if stats is None:
            for key, value in aggregate.items():
                if key.startswith(metric_name) or metric_name.startswith(key):
                    stats = value
                    matched_metric = key
                    break

        return matched_metric, stats

    def _canonical_target(self, target: str) -> str:
        return target.strip().lower()

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _prepare_binary(self, name: str, source: str) -> PreparedBinary:
        source_path = self.source_dir / f"{name}.cu"
        binary_path = self.binary_dir / name
        source_hash = self._hash_text(source)

        existing = self._compiled.get(name)
        if existing and existing.source_hash == source_hash and existing.binary_path.exists():
            return existing

        should_write = True
        if source_path.exists():
            old_hash = self._hash_text(source_path.read_text(encoding="utf-8"))
            should_write = old_hash != source_hash

        if should_write:
            source_path.write_text(source, encoding="utf-8")

        compile_cuda_source(
            source_path,
            binary_path,
            timeout_s=self.config.compile_timeout_s,
        )

        prepared = PreparedBinary(
            source_path=source_path,
            binary_path=binary_path,
            source_hash=source_hash,
        )
        self._compiled[name] = prepared
        return prepared

    def _run_binary_json(self, prepared: PreparedBinary, args: list[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        result = run_binary(
            prepared.binary_path,
            args=args,
            timeout_s=self.config.run_timeout_s,
        )
        payload = extract_last_json_object(result.stdout)
        evidence = {
            "command": " ".join([str(prepared.binary_path), *(args or [])]),
            "duration_seconds": result.duration_seconds,
            "stdout_tail": result.stdout[-1200:],
            "stderr_tail": result.stderr[-1200:],
        }
        return payload, evidence

    def _profile_evidence(self, prepared: PreparedBinary, metrics: list[str]) -> dict[str, Any]:
        try:
            prof = profile_with_ncu(
                prepared.binary_path,
                args=None,
                metrics=metrics,
                timeout_s=self.config.profile_timeout_s,
            )
            records = parse_ncu_csv(prof.stdout)
            return {
                "duration_seconds": prof.duration_seconds,
                "record_count": len(records),
                "aggregate": aggregate_metric_records(records),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _collect_trials(
        self,
        prepared: PreparedBinary,
        args_provider: Callable[[int], list[str] | None],
        value_key: str,
        trials: int | None = None,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        total_trials = trials if trials is not None else self.config.max_trials
        values: list[float] = []
        evidences: list[dict[str, Any]] = []

        for trial in range(total_trials):
            payload, evidence = self._run_binary_json(prepared, args=args_provider(trial))
            if value_key not in payload:
                raise RuntimeError(f"Probe output missing key '{value_key}': {json.dumps(payload)}")
            value = float(payload[value_key])
            values.append(value)
            evidence["payload"] = payload
            evidences.append(evidence)

        return values, evidences

    def _build_result(
        self,
        *,
        target: str,
        value: float,
        unit: str,
        method: str,
        trials: list[float],
        evidence: dict[str, Any],
    ) -> ProbeResult:
        summary = summarize_trials(trials)
        return ProbeResult(
            target=target,
            status="ok",
            value=float(value),
            unit=unit,
            method=method,
            confidence=estimate_confidence(summary),
            trials=trials,
            summary=summary,
            evidence=evidence,
        )

    # --------------------
    # Concrete probes
    # --------------------

    def _probe_physical_sm_count(self, original_target: str) -> ProbeResult:
        prepared = self._prepare_binary("probe_physical_sm_count", _source_sm_count())
        values, evidences = self._collect_trials(
            prepared,
            args_provider=lambda _trial: None,
            value_key="sm_count",
            trials=max(1, self.config.max_trials),
        )
        summary = summarize_trials(values)
        value = int(round(summary["median"]))

        return ProbeResult(
            target=original_target,
            status="ok",
            value=float(value),
            unit="count",
            method="smid-discovery-kernel",
            confidence=estimate_confidence(summary),
            trials=values,
            summary=summary,
            evidence={"runs": evidences},
        )

    def _probe_actual_core_clock_mhz(self, original_target: str) -> ProbeResult:
        prepared = self._prepare_binary("probe_core_clock_mhz", _source_core_clock())
        values, evidences = self._collect_trials(
            prepared,
            args_provider=lambda _trial: ["8000000", "2048", "256"],
            value_key="clock_mhz",
        )
        summary = summarize_trials(values)
        ncu = self._profile_evidence(
            prepared,
            metrics=[
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
            ],
        )

        return self._build_result(
            target=original_target,
            value=summary["median"],
            unit="MHz",
            method="clock64-over-elapsed-time",
            trials=values,
            evidence={"runs": evidences, "ncu": ncu},
        )

    def _probe_peak_fp32_tflops(self, original_target: str) -> ProbeResult:
        prepared = self._prepare_binary("probe_peak_fp32", _source_peak_fp32())
        values, evidences = self._collect_trials(
            prepared,
            args_provider=lambda _trial: ["50000", "2048", "256"],
            value_key="tflops",
        )
        summary = summarize_trials(values)
        ncu = self._profile_evidence(
            prepared,
            metrics=[
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
                "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
            ],
        )

        return self._build_result(
            target=original_target,
            value=summary["median"],
            unit="TFLOP/s",
            method="counted-fma-operations",
            trials=values,
            evidence={"runs": evidences, "ncu": ncu},
        )

    def _probe_l1_latency_cycles(self, original_target: str) -> ProbeResult:
        return self._probe_pointer_chase_latency(original_target, size_kb=16, method="pointer-chase-l1")

    def _probe_l2_latency_cycles(self, original_target: str) -> ProbeResult:
        return self._probe_pointer_chase_latency(original_target, size_kb=1024, method="pointer-chase-l2")

    def _probe_dram_latency_cycles(self, original_target: str) -> ProbeResult:
        return self._probe_pointer_chase_latency(original_target, size_kb=65536, method="pointer-chase-dram")

    def _probe_pointer_chase_latency(self, original_target: str, *, size_kb: int, method: str) -> ProbeResult:
        prepared = self._prepare_binary("probe_pointer_chase", _source_pointer_chase())
        values, evidences = self._collect_trials(
            prepared,
            args_provider=lambda _trial: [str(size_kb), "4000000"],
            value_key="cycles_per_access",
        )
        summary = summarize_trials(values)

        return self._build_result(
            target=original_target,
            value=summary["median"],
            unit="cycles",
            method=method,
            trials=values,
            evidence={"working_set_kb": size_kb, "runs": evidences},
        )

    def _probe_l2_cache_capacity_kb(self, original_target: str) -> ProbeResult:
        prepared = self._prepare_binary("probe_pointer_chase", _source_pointer_chase())
        sizes_kb = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        latencies: list[tuple[int, float]] = []
        runs: list[dict[str, Any]] = []

        for size_kb in sizes_kb:
            payload, evidence = self._run_binary_json(prepared, args=[str(size_kb), "2500000"])
            if "cycles_per_access" not in payload:
                raise RuntimeError(f"Pointer chase output missing cycles_per_access: {payload}")
            latency = float(payload["cycles_per_access"])
            latencies.append((size_kb, latency))
            evidence["payload"] = payload
            runs.append(evidence)

        baseline = sum(lat for _size, lat in latencies[:3]) / 3.0
        cliff_at = None
        for i in range(1, len(latencies)):
            size_kb, latency = latencies[i]
            _prev_size, prev_latency = latencies[i - 1]
            if latency > baseline * 1.8 and latency > prev_latency * 1.25:
                cliff_at = size_kb
                break

        if cliff_at is None:
            estimate = float(latencies[-1][0])
        else:
            idx = max(0, sizes_kb.index(cliff_at) - 1)
            estimate = float(sizes_kb[idx])

        synthetic_trials = [estimate]
        summary = summarize_trials(synthetic_trials)

        return ProbeResult(
            target=original_target,
            status="ok",
            value=estimate,
            unit="KB",
            method="latency-cliff-scan",
            confidence=min(0.85, estimate_confidence(summary)),
            trials=synthetic_trials,
            summary=summary,
            evidence={"latency_curve": latencies, "runs": runs},
        )

    def _probe_global_bandwidth_gbps(self, original_target: str) -> ProbeResult:
        prepared = self._prepare_binary("probe_global_bw", _source_global_bandwidth())
        values, evidences = self._collect_trials(
            prepared,
            args_provider=lambda _trial: ["16777216", "20", "1024", "256"],
            value_key="gbps",
        )
        summary = summarize_trials(values)

        return self._build_result(
            target=original_target,
            value=summary["median"],
            unit="GB/s",
            method="global-copy-throughput",
            trials=values,
            evidence={"runs": evidences},
        )

    def _probe_shared_bandwidth_gbps(self, original_target: str) -> ProbeResult:
        prepared = self._prepare_binary("probe_shared_bw", _source_shared_bandwidth())
        values, evidences = self._collect_trials(
            prepared,
            args_provider=lambda _trial: ["4000000", "1024", "256"],
            value_key="gbps",
        )
        summary = summarize_trials(values)

        return self._build_result(
            target=original_target,
            value=summary["median"],
            unit="GB/s",
            method="shared-memory-loop-throughput",
            trials=values,
            evidence={"runs": evidences},
        )

    def _probe_bank_conflict_penalty_cycles(self, original_target: str) -> ProbeResult:
        prepared = self._prepare_binary("probe_bank_conflict", _source_bank_conflict())
        values, evidences = self._collect_trials(
            prepared,
            args_provider=lambda _trial: ["20000000"],
            value_key="penalty_cycles",
        )
        summary = summarize_trials(values)

        return self._build_result(
            target=original_target,
            value=summary["median"],
            unit="cycles",
            method="shared-bank-conflict-delta",
            trials=values,
            evidence={"runs": evidences},
        )

    def _probe_max_shmem_per_block_kb(self, original_target: str) -> ProbeResult:
        prepared = self._prepare_binary("probe_max_shmem", _source_max_shmem())
        values, evidences = self._collect_trials(
            prepared,
            args_provider=lambda _trial: None,
            value_key="max_shmem_kb",
            trials=1,
        )
        summary = summarize_trials(values)
        rounded = float(math.floor(summary["median"]))

        return ProbeResult(
            target=original_target,
            status="ok",
            value=rounded,
            unit="KB",
            method="dynamic-shared-memory-binary-search",
            confidence=0.8,
            trials=values,
            summary=summary,
            evidence={"runs": evidences},
        )


def _source_sm_count() -> str:
    return r'''
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <set>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            printf("{\"error\":\"%s\"}\n", cudaGetErrorString(err));      \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__device__ __forceinline__ unsigned int read_smid() {
    unsigned int smid;
    asm volatile("mov.u32 %0, %smid;" : "=r"(smid));
    return smid;
}

__global__ void collect_smids(unsigned int* out, int rounds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sm = read_smid();
    unsigned int mix = idx ^ sm;

    for (int i = 0; i < rounds; ++i) {
        mix = mix * 1664525u + 1013904223u;
    }

    out[idx] = sm;
    if (mix == 0u) {
        out[idx] = sm;
    }
}

int main() {
    const int blocks = 4096;
    const int threads = 128;
    const int total = blocks * threads;

    unsigned int* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(unsigned int) * total));

    collect_smids<<<blocks, threads>>>(d_out, 256);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<unsigned int> h_out(total);
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, sizeof(unsigned int) * total, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_out));

    std::set<unsigned int> unique_smids;
    for (unsigned int sm : h_out) {
        unique_smids.insert(sm);
    }

    printf("{\"sm_count\":%zu}\n", unique_smids.size());
    return 0;
}
'''


def _source_core_clock() -> str:
    return r'''
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            printf("{\"error\":\"%s\"}\n", cudaGetErrorString(err));      \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void warmup_kernel(float* sink, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float x = static_cast<float>(idx + 1);
    for (int i = 0; i < iters; ++i) {
        x = fmaf(x, 1.000001f, 0.000001f);
        x = fmaf(x, 0.999999f, 0.000002f);
    }
    if (idx == 0) {
        sink[0] = x;
    }
}

__global__ void clock_kernel(unsigned long long* cycles, float* sink, int iters) {
    float x = static_cast<float>(threadIdx.x + 1);
    unsigned long long start = clock64();

    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        x = fmaf(x, 1.000001f, 0.000001f);
        x = fmaf(x, 0.999999f, 0.000002f);
        x = fmaf(x, 1.000003f, 0.000003f);
        x = fmaf(x, 0.999997f, 0.000004f);
    }

    unsigned long long end = clock64();
    if (threadIdx.x == 0) {
        cycles[0] = end - start;
        sink[0] = x;
    }
}

int main(int argc, char** argv) {
    int iters = 8000000;
    int warm_blocks = 2048;
    int warm_threads = 256;
    if (argc > 1) {
        iters = atoi(argv[1]);
    }
    if (argc > 2) {
        warm_blocks = atoi(argv[2]);
    }
    if (argc > 3) {
        warm_threads = atoi(argv[3]);
    }

    float* d_sink = nullptr;
    unsigned long long* d_cycles = nullptr;
    CHECK_CUDA(cudaMalloc(&d_sink, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(unsigned long long)));

    warmup_kernel<<<warm_blocks, warm_threads>>>(d_sink, 4000);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start_ev, stop_ev;
    CHECK_CUDA(cudaEventCreate(&start_ev));
    CHECK_CUDA(cudaEventCreate(&stop_ev));

    CHECK_CUDA(cudaEventRecord(start_ev));
    clock_kernel<<<1, 256>>>(d_cycles, d_sink, iters);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop_ev));
    CHECK_CUDA(cudaEventSynchronize(stop_ev));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev));

    unsigned long long h_cycles = 0;
    CHECK_CUDA(cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventDestroy(start_ev));
    CHECK_CUDA(cudaEventDestroy(stop_ev));
    CHECK_CUDA(cudaFree(d_sink));
    CHECK_CUDA(cudaFree(d_cycles));

    double mhz = 0.0;
    if (elapsed_ms > 0.0f) {
        mhz = static_cast<double>(h_cycles) / (static_cast<double>(elapsed_ms) * 1000.0);
    }

    printf("{\"clock_mhz\":%.6f,\"elapsed_ms\":%.6f,\"cycles\":%llu}\n", mhz, elapsed_ms, h_cycles);
    return 0;
}
'''


def _source_peak_fp32() -> str:
    return r'''
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            printf("{\"error\":\"%s\"}\n", cudaGetErrorString(err));      \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void fp32_stress(float* out, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float a0 = idx * 0.001f + 1.0f;
    float a1 = a0 + 0.1f;
    float a2 = a1 + 0.2f;
    float a3 = a2 + 0.3f;
    float a4 = a3 + 0.4f;
    float a5 = a4 + 0.5f;
    float a6 = a5 + 0.6f;
    float a7 = a6 + 0.7f;

    #pragma unroll 1
    for (int i = 0; i < iters; ++i) {
        a0 = fmaf(a0, 1.000001f, a1);
        a1 = fmaf(a1, 0.999999f, a2);
        a2 = fmaf(a2, 1.000002f, a3);
        a3 = fmaf(a3, 0.999998f, a4);
        a4 = fmaf(a4, 1.000003f, a5);
        a5 = fmaf(a5, 0.999997f, a6);
        a6 = fmaf(a6, 1.000004f, a7);
        a7 = fmaf(a7, 0.999996f, a0);
    }

    out[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

int main(int argc, char** argv) {
    int iters = 50000;
    int blocks = 2048;
    int threads = 256;
    if (argc > 1) {
        iters = atoi(argv[1]);
    }
    if (argc > 2) {
        blocks = atoi(argv[2]);
    }
    if (argc > 3) {
        threads = atoi(argv[3]);
    }

    const int total_threads = blocks * threads;
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float) * static_cast<size_t>(total_threads)));

    fp32_stress<<<blocks, threads>>>(d_out, 2000);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start_ev, stop_ev;
    CHECK_CUDA(cudaEventCreate(&start_ev));
    CHECK_CUDA(cudaEventCreate(&stop_ev));

    CHECK_CUDA(cudaEventRecord(start_ev));
    fp32_stress<<<blocks, threads>>>(d_out, iters);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop_ev));
    CHECK_CUDA(cudaEventSynchronize(stop_ev));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev));

    CHECK_CUDA(cudaEventDestroy(start_ev));
    CHECK_CUDA(cudaEventDestroy(stop_ev));
    CHECK_CUDA(cudaFree(d_out));

    double flops = static_cast<double>(total_threads) * static_cast<double>(iters) * 16.0;
    double elapsed_s = static_cast<double>(elapsed_ms) / 1000.0;
    double tflops = elapsed_s > 0.0 ? (flops / elapsed_s) / 1e12 : 0.0;

    printf("{\"tflops\":%.6f,\"elapsed_ms\":%.6f,\"flops\":%.0f}\n", tflops, elapsed_ms, flops);
    return 0;
}
'''


def _source_pointer_chase() -> str:
    return r'''
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            printf("{\"error\":\"%s\"}\n", cudaGetErrorString(err));      \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void pointer_chase_kernel(const int* next, int steps, unsigned long long* cycles_out, int* sink) {
    int idx = 0;
    unsigned long long start = clock64();

    #pragma unroll 1
    for (int i = 0; i < steps; ++i) {
        idx = next[idx];
    }

    unsigned long long end = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cycles_out[0] = end - start;
        sink[0] = idx;
    }
}

int main(int argc, char** argv) {
    int size_kb = 1024;
    int steps = 4000000;

    if (argc > 1) {
        size_kb = atoi(argv[1]);
    }
    if (argc > 2) {
        steps = atoi(argv[2]);
    }

    int element_count = (size_kb * 1024) / static_cast<int>(sizeof(int));
    if (element_count < 1024) {
        element_count = 1024;
    }

    std::vector<int> host_next(static_cast<size_t>(element_count));
    const int stride = 32;
    for (int i = 0; i < element_count; ++i) {
        int next = i + stride;
        if (next >= element_count) {
            next -= element_count;
        }
        host_next[static_cast<size_t>(i)] = next;
    }

    int* d_next = nullptr;
    int* d_sink = nullptr;
    unsigned long long* d_cycles = nullptr;

    CHECK_CUDA(cudaMalloc(&d_next, sizeof(int) * static_cast<size_t>(element_count)));
    CHECK_CUDA(cudaMalloc(&d_sink, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemcpy(d_next, host_next.data(), sizeof(int) * static_cast<size_t>(element_count), cudaMemcpyHostToDevice));

    pointer_chase_kernel<<<1, 1>>>(d_next, 50000, d_cycles, d_sink);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    pointer_chase_kernel<<<1, 1>>>(d_next, steps, d_cycles, d_sink);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    unsigned long long cycles = 0;
    CHECK_CUDA(cudaMemcpy(&cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_next));
    CHECK_CUDA(cudaFree(d_sink));
    CHECK_CUDA(cudaFree(d_cycles));

    double cycles_per_access = steps > 0 ? static_cast<double>(cycles) / static_cast<double>(steps) : 0.0;
    printf("{\"size_kb\":%d,\"cycles_per_access\":%.6f,\"steps\":%d}\n", size_kb, cycles_per_access, steps);
    return 0;
}
'''


def _source_global_bandwidth() -> str:
    return r'''
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            printf("{\"error\":\"%s\"}\n", cudaGetErrorString(err));      \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void copy_kernel(float* dst, const float* src, int n, int repeat) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int r = 0; r < repeat; ++r) {
        for (int i = tid; i < n; i += stride) {
            float v = src[i];
            dst[i] = v + 1.0f;
        }
    }
}

int main(int argc, char** argv) {
    int n = 1 << 24;
    int repeat = 20;
    int blocks = 1024;
    int threads = 256;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) repeat = atoi(argv[2]);
    if (argc > 3) blocks = atoi(argv[3]);
    if (argc > 4) threads = atoi(argv[4]);

    float* d_src = nullptr;
    float* d_dst = nullptr;

    CHECK_CUDA(cudaMalloc(&d_src, sizeof(float) * static_cast<size_t>(n)));
    CHECK_CUDA(cudaMalloc(&d_dst, sizeof(float) * static_cast<size_t>(n)));

    CHECK_CUDA(cudaMemset(d_src, 0, sizeof(float) * static_cast<size_t>(n)));
    CHECK_CUDA(cudaMemset(d_dst, 0, sizeof(float) * static_cast<size_t>(n)));

    copy_kernel<<<blocks, threads>>>(d_dst, d_src, n, 2);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start_ev, stop_ev;
    CHECK_CUDA(cudaEventCreate(&start_ev));
    CHECK_CUDA(cudaEventCreate(&stop_ev));

    CHECK_CUDA(cudaEventRecord(start_ev));
    copy_kernel<<<blocks, threads>>>(d_dst, d_src, n, repeat);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop_ev));
    CHECK_CUDA(cudaEventSynchronize(stop_ev));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev));

    CHECK_CUDA(cudaEventDestroy(start_ev));
    CHECK_CUDA(cudaEventDestroy(stop_ev));
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));

    double elapsed_s = static_cast<double>(elapsed_ms) / 1000.0;
    double total_bytes = static_cast<double>(n) * sizeof(float) * 2.0 * static_cast<double>(repeat);
    double gbps = elapsed_s > 0.0 ? (total_bytes / elapsed_s) / 1e9 : 0.0;

    printf("{\"gbps\":%.6f,\"elapsed_ms\":%.6f,\"bytes\":%.0f}\n", gbps, elapsed_ms, total_bytes);
    return 0;
}
'''


def _source_shared_bandwidth() -> str:
    return r'''
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            printf("{\"error\":\"%s\"}\n", cudaGetErrorString(err));      \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void shared_bw_kernel(float* sink, int repeat) {
    __shared__ float tile[1024];
    int tid = threadIdx.x;

    for (int i = tid; i < 1024; i += blockDim.x) {
        tile[i] = static_cast<float>(i);
    }
    __syncthreads();

    float x = static_cast<float>(tid + 1);

    for (int r = 0; r < repeat; ++r) {
        int read_idx = (tid + r) & 1023;
        int write_idx = (tid * 7 + r) & 1023;
        x += tile[read_idx];
        tile[write_idx] = x;
    }

    sink[blockIdx.x * blockDim.x + tid] = x;
}

int main(int argc, char** argv) {
    int repeat = 4000000;
    int blocks = 1024;
    int threads = 256;

    if (argc > 1) repeat = atoi(argv[1]);
    if (argc > 2) blocks = atoi(argv[2]);
    if (argc > 3) threads = atoi(argv[3]);

    int total_threads = blocks * threads;
    float* d_sink = nullptr;
    CHECK_CUDA(cudaMalloc(&d_sink, sizeof(float) * static_cast<size_t>(total_threads)));

    shared_bw_kernel<<<blocks, threads>>>(d_sink, 1000);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start_ev, stop_ev;
    CHECK_CUDA(cudaEventCreate(&start_ev));
    CHECK_CUDA(cudaEventCreate(&stop_ev));

    CHECK_CUDA(cudaEventRecord(start_ev));
    shared_bw_kernel<<<blocks, threads>>>(d_sink, repeat);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop_ev));
    CHECK_CUDA(cudaEventSynchronize(stop_ev));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev));

    CHECK_CUDA(cudaEventDestroy(start_ev));
    CHECK_CUDA(cudaEventDestroy(stop_ev));
    CHECK_CUDA(cudaFree(d_sink));

    double elapsed_s = static_cast<double>(elapsed_ms) / 1000.0;
    double total_bytes = static_cast<double>(total_threads) * static_cast<double>(repeat) * 8.0;
    double gbps = elapsed_s > 0.0 ? (total_bytes / elapsed_s) / 1e9 : 0.0;

    printf("{\"gbps\":%.6f,\"elapsed_ms\":%.6f,\"bytes\":%.0f}\n", gbps, elapsed_ms, total_bytes);
    return 0;
}
'''


def _source_bank_conflict() -> str:
    return r'''
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            printf("{\"error\":\"%s\"}\n", cudaGetErrorString(err));      \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void bank_probe(unsigned long long* out_free, unsigned long long* out_conflict, int iters) {
    __shared__ float shared[1024];
    int tid = threadIdx.x;

    shared[tid] = static_cast<float>(tid + 1);
    __syncthreads();

    float x = static_cast<float>(tid + 1);

    unsigned long long start_free = clock64();
    for (int i = 0; i < iters; ++i) {
        x += shared[(tid + i) & 31];
    }
    unsigned long long end_free = clock64();

    unsigned long long start_conflict = clock64();
    for (int i = 0; i < iters; ++i) {
        int idx = (tid * 32 + i) & 1023;
        x += shared[idx];
    }
    unsigned long long end_conflict = clock64();

    if (tid == 0) {
        out_free[0] = end_free - start_free;
        out_conflict[0] = end_conflict - start_conflict;
    }

    if (x == 0.0f) {
        shared[0] = x;
    }
}

int main(int argc, char** argv) {
    int iters = 20000000;
    if (argc > 1) {
        iters = atoi(argv[1]);
    }

    unsigned long long* d_free = nullptr;
    unsigned long long* d_conflict = nullptr;
    CHECK_CUDA(cudaMalloc(&d_free, sizeof(unsigned long long)));
    CHECK_CUDA(cudaMalloc(&d_conflict, sizeof(unsigned long long)));

    bank_probe<<<1, 32>>>(d_free, d_conflict, iters);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    unsigned long long h_free = 0;
    unsigned long long h_conflict = 0;
    CHECK_CUDA(cudaMemcpy(&h_free, d_free, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_conflict, d_conflict, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_free));
    CHECK_CUDA(cudaFree(d_conflict));

    double penalty = iters > 0 ? (static_cast<double>(h_conflict) - static_cast<double>(h_free)) / static_cast<double>(iters) : 0.0;
    printf("{\"free_cycles\":%llu,\"conflict_cycles\":%llu,\"penalty_cycles\":%.6f}\n", h_free, h_conflict, penalty);
    return 0;
}
'''


def _source_max_shmem() -> str:
    return r'''
#include <cuda_runtime.h>

#include <cstdio>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            printf("{\"error\":\"%s\"}\n", cudaGetErrorString(err));      \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void dummy_kernel(float* out) {
    if (threadIdx.x == 0) {
        out[0] = 1.0f;
    }
}

bool launch_with_smem(int shared_bytes, float* d_out) {
    dummy_kernel<<<1, 1, shared_bytes>>>(d_out);
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    cudaError_t sync_err = cudaDeviceSynchronize();
    if (sync_err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    return true;
}

int main() {
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));

    int low_kb = 0;
    int high_kb = 256;

    while (low_kb < high_kb) {
        int mid_kb = (low_kb + high_kb + 1) / 2;
        int bytes = mid_kb * 1024;
        if (launch_with_smem(bytes, d_out)) {
            low_kb = mid_kb;
        } else {
            high_kb = mid_kb - 1;
        }
    }

    CHECK_CUDA(cudaFree(d_out));
    printf("{\"max_shmem_kb\":%d}\n", low_kb);
    return 0;
}
'''


def _source_device_attribute() -> str:
    return r'''
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            printf("{\"error\":\"%s\"}\n", cudaGetErrorString(err));      \
            return 1;                                                          \
        }                                                                      \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("{\"error\":\"missing attribute name\"}\n");
        return 1;
    }

    const char* attr = argv[1];
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    if (strcmp(attr, "fb_bus_width") == 0) {
        printf("{\"value\":%d,\"unit\":\"bits\"}\n", prop.memoryBusWidth);
        return 0;
    }

    if (strcmp(attr, "max_gpu_frequency_khz") == 0) {
        printf("{\"value\":%d,\"unit\":\"kHz\"}\n", prop.clockRate);
        return 0;
    }

    if (strcmp(attr, "max_mem_frequency_khz") == 0) {
        printf("{\"value\":%d,\"unit\":\"kHz\"}\n", prop.memoryClockRate);
        return 0;
    }

    if (strcmp(attr, "sm_count") == 0 || strcmp(attr, "multi_processor_count") == 0) {
        printf("{\"value\":%d,\"unit\":\"count\"}\n", prop.multiProcessorCount);
        return 0;
    }

    printf("{\"error\":\"unsupported attribute\",\"attribute\":\"%s\"}\n", attr);
    return 1;
}
'''


def _source_metric_workload() -> str:
    return r'''
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            printf("{\"error\":\"%s\"}\n", cudaGetErrorString(err));      \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void metric_workload(float* out, const float* in, int n, int repeat) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int r = 0; r < repeat; ++r) {
        for (int i = tid; i < n; i += stride) {
            float v = in[i];
            v = fmaf(v, 1.000001f, 0.000001f);
            v = fmaf(v, 0.999999f, 0.000002f);
            out[i] = v;
        }
    }
}

int main(int argc, char** argv) {
    int n = 1 << 24;
    int repeat = 30;
    int blocks = 1024;
    int threads = 256;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) repeat = atoi(argv[2]);
    if (argc > 3) blocks = atoi(argv[3]);
    if (argc > 4) threads = atoi(argv[4]);

    float* d_in = nullptr;
    float* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, sizeof(float) * static_cast<size_t>(n)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float) * static_cast<size_t>(n)));
    CHECK_CUDA(cudaMemset(d_in, 0, sizeof(float) * static_cast<size_t>(n)));

    metric_workload<<<blocks, threads>>>(d_out, d_in, n, 2);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start_ev, stop_ev;
    CHECK_CUDA(cudaEventCreate(&start_ev));
    CHECK_CUDA(cudaEventCreate(&stop_ev));
    CHECK_CUDA(cudaEventRecord(start_ev));

    metric_workload<<<blocks, threads>>>(d_out, d_in, n, repeat);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop_ev));
    CHECK_CUDA(cudaEventSynchronize(stop_ev));
    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev));

    CHECK_CUDA(cudaEventDestroy(start_ev));
    CHECK_CUDA(cudaEventDestroy(stop_ev));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    printf("{\"elapsed_ms\":%.6f}\n", elapsed_ms);
    return 0;
}
'''
