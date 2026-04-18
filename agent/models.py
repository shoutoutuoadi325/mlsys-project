from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TargetSpec:
    targets: list[str]
    raw: dict[str, Any]


@dataclass
class ProbeResult:
    target: str
    status: str
    value: float | None
    unit: str
    method: str
    confidence: float
    trials: list[float] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class TargetPlan:
    target: str
    kind: str
    source: str
    rationale: str
    probe_name: str | None = None
    attribute_name: str | None = None
    metric_name: str | None = None
    unit_hint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "kind": self.kind,
            "source": self.source,
            "rationale": self.rationale,
            "probe_name": self.probe_name,
            "attribute_name": self.attribute_name,
            "metric_name": self.metric_name,
            "unit_hint": self.unit_hint,
        }


@dataclass
class AgentRunSummary:
    student_id: str
    started_at: str
    finished_at: str
    duration_seconds: float
    target_count: int
    success_count: int
    failure_count: int
