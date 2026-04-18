from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent.config import AgentConfig
from agent.models import TargetPlan
from llm.openai_client import get_openai_client


class TargetPlanner:
    def __init__(self, config: AgentConfig, builtin_probe_names: list[str]):
        self._config = config
        self._builtin_probe_names = set(builtin_probe_names)

    def plan_target(self, target: str) -> TargetPlan:
        llm_plan = self._plan_with_llm(target)
        if llm_plan is not None:
            return llm_plan
        return self._fallback_plan(target)

    def _plan_with_llm(self, target: str) -> TargetPlan | None:
        client = get_openai_client()
        model = os.getenv("BASE_MODEL", "").strip()
        if client is None or not model:
            return None

        prompt_file = self._config.prompt_dir / "plan_target.txt"
        if not prompt_file.exists():
            return None

        prompt_template = prompt_file.read_text(encoding="utf-8")
        prompt = prompt_template.format(
            target=target,
            builtin_probe_names=", ".join(sorted(self._builtin_probe_names)),
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            return None

        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            return None

        payload = self._extract_json_payload(content)
        if payload is None:
            return None

        plan = self._validate_plan_payload(target, payload)
        if plan is None:
            return None
        return plan

    def _extract_json_payload(self, text: str) -> dict[str, Any] | None:
        stripped = text.strip()
        try:
            value = json.loads(stripped)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

        first = stripped.find("{")
        last = stripped.rfind("}")
        if first == -1 or last == -1 or last <= first:
            return None

        candidate = stripped[first : last + 1]
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            return None
        return None

    def _validate_plan_payload(self, target: str, payload: dict[str, Any]) -> TargetPlan | None:
        kind = str(payload.get("kind", "")).strip()
        if kind not in {"builtin_probe", "device_attribute", "ncu_metric"}:
            return None

        rationale = str(payload.get("rationale", "")).strip() or "planned by llm"
        unit_hint = str(payload.get("unit_hint", "")).strip()

        probe_name = str(payload.get("probe_name", "")).strip() or None
        attribute_name = str(payload.get("attribute_name", "")).strip() or None
        metric_name = str(payload.get("metric_name", "")).strip() or None

        if kind == "builtin_probe":
            if probe_name is None or probe_name not in self._builtin_probe_names:
                return None
        elif kind == "device_attribute":
            if attribute_name is None:
                return None
        elif kind == "ncu_metric":
            if metric_name is None:
                return None

        return TargetPlan(
            target=target,
            kind=kind,
            source="llm",
            rationale=rationale,
            probe_name=probe_name,
            attribute_name=attribute_name,
            metric_name=metric_name,
            unit_hint=unit_hint,
        )

    def _fallback_plan(self, target: str) -> TargetPlan:
        canonical = target.strip().lower()

        if canonical in self._builtin_probe_names:
            return TargetPlan(
                target=target,
                kind="builtin_probe",
                source="fallback",
                rationale="target name matches built-in probe",
                probe_name=canonical,
                unit_hint=self._infer_unit_hint(canonical),
            )

        if canonical in {"launch__sm_count", "physical_sm_count"}:
            return TargetPlan(
                target=target,
                kind="builtin_probe",
                source="fallback",
                rationale="launch sm count maps to physical SM probe",
                probe_name="physical_sm_count",
                unit_hint="count",
            )

        if canonical.startswith("device__attribute_"):
            attr = canonical.replace("device__attribute_", "", 1)
            return TargetPlan(
                target=target,
                kind="device_attribute",
                source="fallback",
                rationale="device attribute prefix detected",
                attribute_name=attr,
                unit_hint=self._infer_unit_hint(canonical),
            )

        if self._looks_like_ncu_metric(canonical):
            return TargetPlan(
                target=target,
                kind="ncu_metric",
                source="fallback",
                rationale="ncu metric naming pattern detected",
                metric_name=target.strip(),
                unit_hint=self._infer_unit_hint(canonical),
            )

        return TargetPlan(
            target=target,
            kind="ncu_metric",
            source="fallback",
            rationale="default to ncu metric probe for unknown target",
            metric_name=target.strip(),
            unit_hint=self._infer_unit_hint(canonical),
        )

    def _looks_like_ncu_metric(self, target: str) -> bool:
        return target.startswith(("sm__", "gpu__", "dram__", "l1tex__", "l2__", "launch__"))

    def _infer_unit_hint(self, target: str) -> str:
        if target.endswith("_khz"):
            return "kHz"
        if target.endswith("_mhz"):
            return "MHz"
        if target.endswith("_count"):
            return "count"
        if "per_second" in target:
            return "per_second"
        if "pct_of_peak" in target:
            return "%"
        return ""
