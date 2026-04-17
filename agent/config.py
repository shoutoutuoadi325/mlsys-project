from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentConfig:
    root_dir: Path
    target_spec_path: Path
    output_path: Path
    state_path: Path
    generated_dir: Path
    compile_timeout_s: int
    run_timeout_s: int
    profile_timeout_s: int
    max_trials: int
    student_id: str


def _resolve_default_output_path(root_dir: Path) -> Path:
    workspace_output = Path("/workspace/output.json")
    if workspace_output.parent.exists() and os.access(workspace_output.parent, os.W_OK):
        return workspace_output
    return root_dir / "output.json"


def _resolve_default_target_spec(root_dir: Path) -> Path:
    preferred = Path("/target/target_spec.json")
    if preferred.exists():
        return preferred

    local = root_dir / "target_spec.json"
    if local.exists():
        return local

    sample = root_dir / "sample" / "mlsys-project" / "target_spec_sample.json"
    return sample


def load_config() -> AgentConfig:
    root_dir = Path(__file__).resolve().parents[1]
    target_spec_path = Path(os.getenv("TARGET_SPEC_PATH", str(_resolve_default_target_spec(root_dir))))
    output_path = Path(os.getenv("OUTPUT_PATH", str(_resolve_default_output_path(root_dir))))
    state_path = Path(os.getenv("STATE_PATH", str(root_dir / ".agent_state.json")))
    generated_dir = Path(os.getenv("GENERATED_DIR", str(root_dir / ".generated")))

    compile_timeout_s = int(os.getenv("COMPILE_TIMEOUT_S", "240"))
    run_timeout_s = int(os.getenv("RUN_TIMEOUT_S", "240"))
    profile_timeout_s = int(os.getenv("PROFILE_TIMEOUT_S", "300"))
    max_trials = int(os.getenv("MAX_TRIALS", "3"))
    student_id = os.getenv("STUDENT_ID", "23302010025")

    return AgentConfig(
        root_dir=root_dir,
        target_spec_path=target_spec_path,
        output_path=output_path,
        state_path=state_path,
        generated_dir=generated_dir,
        compile_timeout_s=compile_timeout_s,
        run_timeout_s=run_timeout_s,
        profile_timeout_s=profile_timeout_s,
        max_trials=max_trials,
        student_id=student_id,
    )
