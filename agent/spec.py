from __future__ import annotations

import json
from pathlib import Path

from agent.models import TargetSpec


def load_target_spec(path: Path) -> TargetSpec:
    if not path.exists():
        raise FileNotFoundError(f"Target spec file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    candidate = raw.get("targets")
    if candidate is None:
        candidate = raw.get("metrics")

    if not isinstance(candidate, list) or not candidate:
        raise ValueError(
            "Target spec must include a non-empty list under 'targets' (preferred) or 'metrics'"
        )

    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidate:
        if not isinstance(item, str):
            raise ValueError(f"Invalid target entry type: {type(item)!r}")
        value = item.strip()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)

    if not normalized:
        raise ValueError("No valid targets found in target spec")

    return TargetSpec(targets=normalized, raw=raw)
