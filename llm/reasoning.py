from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from llm.openai_client import get_openai_client


def _fallback_reasoning(results_payload: dict[str, Any]) -> str:
    results = results_payload.get("results", {})
    if not isinstance(results, dict):
        return "No valid numeric results were produced."

    lines: list[str] = []
    for name, value in results.items():
        lines.append(f"- {name}: {value}")

    if not lines:
        return "No successful probes were produced in this run."

    return "Probe summary:\n" + "\n".join(lines)


def maybe_generate_reasoning(results_payload: dict[str, Any]) -> str:
    client = get_openai_client()
    model = os.getenv("BASE_MODEL", "").strip()
    if client is None or not model:
        return _fallback_reasoning(results_payload)

    prompt_file = Path(__file__).resolve().parents[1] / "agent" / "prompts" / "summarize_results.txt"
    if prompt_file.exists():
        prompt_template = prompt_file.read_text(encoding="utf-8")
        prompt = prompt_template.format(
            payload_json=json.dumps(results_payload, indent=2, sort_keys=True)
        )
    else:
        prompt = (
            "You are summarizing GPU hardware probe results. "
            "Write a concise engineering summary with 3 sections: "
            "(1) key findings, (2) confidence caveats, (3) next probes.\n\n"
            f"Results JSON:\n{json.dumps(results_payload, indent=2, sort_keys=True)}"
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if isinstance(content, str) and content.strip():
            return content.strip()
    except Exception:
        return _fallback_reasoning(results_payload)

    return _fallback_reasoning(results_payload)
