from __future__ import annotations

import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def monotonic_seconds() -> float:
    return time.monotonic()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    temp_path.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(text)
    temp_path.replace(path)


def extract_last_json_object(text: str) -> dict[str, Any]:
    start = text.rfind("{")
    while start != -1:
        candidate = text[start:].strip()
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        start = text.rfind("{", 0, start)
    raise ValueError("No JSON object found in command output")


def summarize_trials(values: list[float]) -> dict[str, float]:
    if not values:
        return {}

    ordered = sorted(values)
    median = statistics.median(ordered)
    mean = statistics.fmean(ordered)
    v_min = ordered[0]
    v_max = ordered[-1]
    stdev = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    spread = 0.0 if median == 0 else (v_max - v_min) / abs(median)
    return {
        "median": median,
        "mean": mean,
        "min": v_min,
        "max": v_max,
        "stdev": stdev,
        "spread_ratio": spread,
    }


def estimate_confidence(summary: dict[str, float]) -> float:
    spread = summary.get("spread_ratio", math.inf)
    if spread <= 0.02:
        return 0.95
    if spread <= 0.05:
        return 0.88
    if spread <= 0.1:
        return 0.78
    if spread <= 0.2:
        return 0.65
    return 0.5
