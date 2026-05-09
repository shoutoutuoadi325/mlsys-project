from __future__ import annotations

import gc
import hashlib
import math
import os
import shutil
import statistics
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from agent.lora_sources import LoraCandidate, candidate_suite, fallback_candidate
from agent.utils import atomic_write_json, atomic_write_text, utc_now_iso


@dataclass
class Phase2Config:
    root_dir: Path
    optimized_path: Path
    generated_dir: Path
    output_path: Path
    report_path: Path
    max_candidates: int
    correctness_dims: list[int]
    benchmark_dims: list[int]
    warmup: int
    iters: int
    student_id: str


@dataclass
class BenchmarkCase:
    d: int
    W: Any
    X: Any
    A: Any
    B: Any
    y_ref: Any
    torch_ms: float


@dataclass
class CandidateResult:
    name: str
    description: str
    status: str
    source_path: str
    compile_seconds: float | None = None
    median_ms_by_d: dict[int, float] = field(default_factory=dict)
    speedup_by_d: dict[int, float] = field(default_factory=dict)
    max_abs_err_by_d: dict[int, float] = field(default_factory=dict)
    rel_l2_err_by_d: dict[int, float] = field(default_factory=dict)
    score: float | None = None
    error: str | None = None


def load_phase2_config() -> Phase2Config:
    root_dir = Path(__file__).resolve().parents[1]
    optimized_path = Path(os.getenv("OPTIMIZED_LORA_PATH", str(root_dir / "optimized_lora.cu")))
    generated_dir = Path(os.getenv("GENERATED_DIR", str(root_dir / ".generated" / "phase2")))
    output_path = Path(os.getenv("OUTPUT_PATH", str(_default_workspace_path(root_dir, "output.json"))))
    report_path = Path(os.getenv("REPORT2_PATH", str(_default_workspace_path(root_dir, "report2.md"))))

    correctness_dims = _parse_int_list(os.getenv("LORA_AGENT_CORRECTNESS_DIMS", "256"), default=[256])
    benchmark_dims = _parse_int_list(
        os.getenv("LORA_AGENT_BENCH_DIMS", "3584,4096,4352,4608"),
        default=[3584, 4096, 4352, 4608],
    )

    return Phase2Config(
        root_dir=root_dir,
        optimized_path=optimized_path,
        generated_dir=generated_dir,
        output_path=output_path,
        report_path=report_path,
        max_candidates=int(os.getenv("LORA_AGENT_MAX_CANDIDATES", "5")),
        correctness_dims=correctness_dims,
        benchmark_dims=benchmark_dims,
        warmup=int(os.getenv("LORA_AGENT_WARMUP", "4")),
        iters=int(os.getenv("LORA_AGENT_ITERS", "10")),
        student_id=os.getenv("STUDENT_ID", "23302010025"),
    )


def _default_workspace_path(root_dir: Path, name: str) -> Path:
    workspace = Path("/workspace")
    if workspace.exists() and os.access(workspace, os.W_OK):
        return workspace / name
    return root_dir / name


def _parse_int_list(raw: str, *, default: list[int]) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values or default


class LoraOptimizationAgent:
    def __init__(self, config: Phase2Config):
        self.config = config
        self.config.generated_dir.mkdir(parents=True, exist_ok=True)
        self.candidate_dir = self.config.generated_dir / "candidates"
        self.build_dir = self.config.generated_dir / "torch_extensions"
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.build_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        started_at = utc_now_iso()
        start = time.monotonic()
        fallback = fallback_candidate()
        atomic_write_text(self.config.optimized_path, fallback.source)

        results: list[CandidateResult] = []
        best: CandidateResult | None = None
        best_name = fallback.name

        environment = self._environment_summary()
        self._cleanup_old_phase_outputs()

        try:
            if not environment["torch_importable"]:
                raise RuntimeError("PyTorch is not importable; wrote fallback source only")

            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available; wrote fallback source only")

            if shutil.which("nvcc") is None:
                raise RuntimeError("nvcc is not available; wrote fallback source only")

            cases = self._prepare_benchmark_cases(torch)
            for candidate in candidate_suite()[: self.config.max_candidates]:
                result = self._evaluate_candidate(candidate, cases)
                results.append(result)
                self._release_cuda(torch)
                if result.status != "ok" or result.score is None:
                    continue
                if best is None or result.score > (best.score or -math.inf):
                    best = result
                    best_name = candidate.name
                    atomic_write_text(self.config.optimized_path, candidate.source)

        except Exception as exc:
            results.append(
                CandidateResult(
                    name="agent_runtime",
                    description="Agent-level runtime guard",
                    status="skipped",
                    source_path="",
                    error=f"{exc}\n{traceback.format_exc()}",
                )
            )

        finished_at = utc_now_iso()
        payload = {
            "student_id": self.config.student_id,
            "phase": 2,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": time.monotonic() - start,
            "environment": environment,
            "optimized_lora_path": str(self.config.optimized_path),
            "best_candidate": best_name,
            "best_score": best.score if best else None,
            "config": {
                "max_candidates": self.config.max_candidates,
                "correctness_dims": self.config.correctness_dims,
                "benchmark_dims": self.config.benchmark_dims,
                "warmup": self.config.warmup,
                "iters": self.config.iters,
            },
            "candidates": [asdict(item) for item in results],
            "methodology": [
                "Generate a known-correct fallback immediately.",
                "Generate multiple single-file CUDA extension candidates from kernel strategy parameters.",
                "Compile candidates with the same PyTorch extension toolchain used by the official harness.",
                "Check each candidate against the PyTorch reference on synthetic tensors.",
                "Benchmark valid candidates with CUDA events and keep the best source in optimized_lora.cu.",
            ],
        }
        atomic_write_json(self.config.output_path, payload)
        self._write_report(payload)
        return payload

    def _environment_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "torch_importable": False,
            "cuda_available": False,
            "nvcc": shutil.which("nvcc"),
        }
        try:
            import torch

            summary["torch_importable"] = True
            summary["torch_version"] = getattr(torch, "__version__", "unknown")
            summary["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                summary["device_name"] = torch.cuda.get_device_name(0)
                summary["device_count"] = torch.cuda.device_count()
        except Exception as exc:
            summary["torch_error"] = str(exc)
        return summary

    def _cleanup_old_phase_outputs(self) -> None:
        target = self.config.output_path.resolve()
        for candidate in self.config.root_dir.glob("output*.json"):
            try:
                if candidate.resolve() != target:
                    candidate.unlink()
            except OSError:
                pass

    def _prepare_benchmark_cases(self, torch: Any) -> list[BenchmarkCase]:
        cases: list[BenchmarkCase] = []
        for d in self.config.benchmark_dims:
            W, X, A, B = self._make_inputs(torch, d, seed=20260508 + d)
            with torch.no_grad():
                y_ref = self._reference_impl(torch, W, X, A, B)
            torch_ms = self._benchmark(lambda: self._reference_impl(torch, W, X, A, B), torch)
            cases.append(BenchmarkCase(d=d, W=W, X=X, A=A, B=B, y_ref=y_ref, torch_ms=torch_ms))
        return cases

    def _evaluate_candidate(self, candidate: LoraCandidate, cases: list[BenchmarkCase]) -> CandidateResult:
        source_hash = hashlib.sha256(candidate.source.encode("utf-8")).hexdigest()[:12]
        source_path = self.candidate_dir / f"{candidate.name}_{source_hash}.cu"
        atomic_write_text(source_path, candidate.source)

        result = CandidateResult(
            name=candidate.name,
            description=candidate.description,
            status="compiling",
            source_path=str(source_path),
        )

        try:
            module, compile_seconds = self._build_module(candidate.name, source_hash, source_path)
            result.compile_seconds = compile_seconds

            import torch

            for d in self.config.correctness_dims:
                W, X, A, B = self._make_inputs(torch, d, seed=10101 + d)
                with torch.no_grad():
                    y_ref = self._reference_impl(torch, W, X, A, B)
                    y = module.forward(W, X, A, B)
                    passed, max_abs, rel_l2 = self._check_correctness(torch, y, y_ref)
                result.max_abs_err_by_d[d] = max_abs
                result.rel_l2_err_by_d[d] = rel_l2
                if not passed:
                    result.status = "failed_correctness"
                    result.error = f"failed correctness at d={d}, max_abs={max_abs}, rel_l2={rel_l2}"
                    return result

            for case in cases:
                with torch.no_grad():
                    y = module.forward(case.W, case.X, case.A, case.B)
                    passed, max_abs, rel_l2 = self._check_correctness(torch, y, case.y_ref)
                result.max_abs_err_by_d[case.d] = max_abs
                result.rel_l2_err_by_d[case.d] = rel_l2
                if not passed:
                    result.status = "failed_correctness"
                    result.error = (
                        f"failed benchmark correctness at d={case.d}, "
                        f"max_abs={max_abs}, rel_l2={rel_l2}"
                    )
                    return result

                median_ms = self._benchmark(
                    lambda c=case: module.forward(c.W, c.X, c.A, c.B),
                    torch,
                )
                result.median_ms_by_d[case.d] = median_ms
                result.speedup_by_d[case.d] = case.torch_ms / median_ms if median_ms > 0 else 0.0

            result.score = self._score(result.speedup_by_d)
            result.status = "ok"
            return result
        except Exception as exc:
            result.status = "failed"
            result.error = f"{exc}\n{traceback.format_exc()}"
            return result

    def _build_module(self, name: str, source_hash: str, source_path: Path) -> tuple[Any, float]:
        from torch.utils.cpp_extension import load

        start = time.monotonic()
        module = load(
            name=f"optimized_lora_{name}_{source_hash}",
            sources=[str(source_path)],
            build_directory=str(self.build_dir),
            verbose=False,
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
        )
        return module, time.monotonic() - start

    def _make_inputs(self, torch: Any, d: int, *, seed: int) -> tuple[Any, Any, Any, Any]:
        torch.manual_seed(seed)
        device = torch.device("cuda")
        W = torch.randn((d, d), device=device, dtype=torch.float32).contiguous()
        X = torch.randn((d, d), device=device, dtype=torch.float32).contiguous()
        A = torch.randn((d, 16), device=device, dtype=torch.float32).contiguous()
        B = torch.randn((d, 16), device=device, dtype=torch.float32).contiguous()
        return W, X, A, B

    def _reference_impl(self, torch: Any, W: Any, X: Any, A: Any, B: Any) -> Any:
        return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)

    def _check_correctness(self, torch: Any, y: Any, y_ref: Any) -> tuple[bool, float, float]:
        diff = (y - y_ref).float()
        max_abs = diff.abs().max().item()
        rel_l2 = (diff.norm() / (y_ref.float().norm() + 1e-12)).item()
        # Large FP32 GEMMs can differ by several ULPs across cuBLAS call shapes.
        # Keep the search guard close to the official tolerance without discarding
        # official-valid no-copy B^T candidates on synthetic tensors.
        passed = bool(rel_l2 <= 1e-5 and max_abs <= 1e-2)
        return passed, float(max_abs), float(rel_l2)

    def _benchmark(self, fn: Callable[[], Any], torch: Any) -> float:
        with torch.no_grad():
            for _ in range(self.config.warmup):
                _ = fn()
            torch.cuda.synchronize()

            times: list[float] = []
            for _ in range(self.config.iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = fn()
                end.record()
                torch.cuda.synchronize()
                times.append(float(start.elapsed_time(end)))
        return float(statistics.median(times))

    def _score(self, speedup_by_d: dict[int, float]) -> float:
        if not speedup_by_d:
            return 0.0
        product = 1.0
        for value in speedup_by_d.values():
            product *= max(value, 1e-9)
        return product ** (1.0 / len(speedup_by_d))

    def _release_cuda(self, torch: Any) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _write_report(self, payload: dict[str, Any]) -> None:
        best = payload.get("best_candidate", "unknown")
        best_score = payload.get("best_score")
        lines = [
            "# Phase 2 Report",
            "",
            f"Student ID: {self.config.student_id}",
            "",
            "## Agent Design",
            "",
            "The agent writes a valid fallback `optimized_lora.cu` before doing any expensive work. "
            "It then generates several CUDA extension candidates for the LoRA operator, compiles them "
            "with the PyTorch extension toolchain, checks correctness against the official-style PyTorch "
            "reference, benchmarks valid candidates with CUDA events, and keeps the best candidate in "
            "`optimized_lora.cu`.",
            "",
            "## Search Space",
            "",
            "- ATen/cuBLAS fallback using `mm` and `addmm`.",
            "- A variant that materializes `B^T` before the rank-16 product to avoid slow strided access cases.",
            "- Precompute variants that form `W + A@B^T` first and then run one large GEMM.",
            "- ATen/cuBLAS for the large products plus custom rank-16 accumulation kernels.",
            "",
            "## Current Best",
            "",
            f"- Best candidate: `{best}`",
            f"- Geometric mean speedup in agent benchmark: `{best_score}`",
            "",
            "## Notes",
            "",
            "The official output id is only available after `/submit2` finishes. "
            "Put the best returned output id into `output_id2.txt` before final report submission.",
            "",
        ]
        atomic_write_text(self.config.report_path, "\n".join(lines))


def run_phase2_agent() -> int:
    config = load_phase2_config()
    payload = LoraOptimizationAgent(config).run()
    print(f"Phase 2 agent completed. optimized_lora.cu: {payload['optimized_lora_path']}")
    print(f"Best candidate: {payload['best_candidate']}")
    print(f"Report: {config.report_path}")
    print(f"Output: {config.output_path}")
    return 0
