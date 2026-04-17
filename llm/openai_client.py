from __future__ import annotations

import os
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime when SDK is absent
    OpenAI = None  # type: ignore


def get_openai_client() -> Any | None:
    if OpenAI is None:
        return None

    api_key = os.getenv("API_KEY", "").strip()
    base_url = os.getenv("BASE_URL", "").strip()
    if not api_key:
        return None

    kwargs: dict[str, str] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)
