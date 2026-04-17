from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.utils import atomic_write_json


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "runs": [],
            "last_error": None,
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            value = json.load(f)
        if isinstance(value, dict):
            value.setdefault("runs", [])
            value.setdefault("last_error", None)
            return value
    except json.JSONDecodeError:
        pass

    return {
        "runs": [],
        "last_error": "State file was corrupted and has been reset",
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    atomic_write_json(path, state)
