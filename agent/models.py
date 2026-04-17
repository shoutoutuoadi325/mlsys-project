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
class AgentRunSummary:
    student_id: str
    started_at: str
    finished_at: str
    duration_seconds: float
    target_count: int
    success_count: int
    failure_count: int
