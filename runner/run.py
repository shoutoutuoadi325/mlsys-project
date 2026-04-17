from __future__ import annotations

import argparse
import csv
import io
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_NCU_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__maximum_warps_per_active_cycle_pct",
]


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


@dataclass
class MetricRecord:
    kernel_name: str
    metric_name: str
    metric_unit: str
    metric_value: float


def _ensure_tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"{name} not found in PATH")
    return resolved


def run_cmd(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout_s: int | None = None,
) -> CommandResult:
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after {timeout_s}s: {' '.join(command)}") from exc

    return CommandResult(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_seconds=time.monotonic() - start,
    )


def compile_cuda_source(
    source_path: Path,
    binary_path: Path,
    *,
    timeout_s: int = 240,
    extra_flags: list[str] | None = None,
) -> CommandResult:
    nvcc = _ensure_tool("nvcc")
    binary_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        nvcc,
        str(source_path),
        "-O3",
        "-std=c++17",
        "-o",
        str(binary_path),
    ]
    if extra_flags:
        command.extend(extra_flags)

    result = run_cmd(command, timeout_s=timeout_s)
    if result.returncode != 0:
        raise RuntimeError(
            "CUDA compilation failed\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def run_binary(
    binary_path: Path,
    args: list[str] | None = None,
    *,
    timeout_s: int = 240,
) -> CommandResult:
    command = [str(binary_path)]
    if args:
        command.extend(args)

    result = run_cmd(command, timeout_s=timeout_s)
    if result.returncode != 0:
        raise RuntimeError(
            "Binary execution failed\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def profile_with_ncu(
    binary_path: Path,
    args: list[str] | None = None,
    *,
    metrics: list[str] | None = None,
    timeout_s: int = 300,
) -> CommandResult:
    ncu = _ensure_tool("ncu")
    metrics_to_collect = metrics or DEFAULT_NCU_METRICS

    command = [
        ncu,
        "-f",
        "--target-processes",
        "all",
        "--metrics",
        ",".join(metrics_to_collect),
        "--csv",
        str(binary_path),
    ]
    if args:
        command.extend(args)

    result = run_cmd(command, timeout_s=timeout_s)
    if result.returncode != 0:
        raise RuntimeError(
            "NCU profiling failed\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def _parse_metric_value(raw: str) -> float | None:
    value = raw.strip().replace(",", "")
    if not value or value.lower() in {"nan", "n/a", "none"}:
        return None

    try:
        return float(value)
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
        if match:
            return float(match.group(0))
        return None


def parse_ncu_csv(csv_text: str) -> list[MetricRecord]:
    rows = list(csv.reader(io.StringIO(csv_text)))
    header: list[str] | None = None
    index_map: dict[str, int] = {}
    records: list[MetricRecord] = []

    for row in rows:
        if not row:
            continue

        if "Metric Name" in row and "Metric Value" in row:
            header = row
            index_map = {name: i for i, name in enumerate(header)}
            continue

        if header is None:
            continue

        if len(row) < len(header):
            continue

        metric_name = row[index_map.get("Metric Name", -1)].strip() if "Metric Name" in index_map else ""
        if not metric_name:
            continue

        metric_value_raw = row[index_map.get("Metric Value", -1)] if "Metric Value" in index_map else ""
        parsed_value = _parse_metric_value(metric_value_raw)
        if parsed_value is None:
            continue

        metric_unit = row[index_map.get("Metric Unit", -1)].strip() if "Metric Unit" in index_map else ""
        kernel_name = row[index_map.get("Kernel Name", -1)].strip() if "Kernel Name" in index_map else ""

        records.append(
            MetricRecord(
                kernel_name=kernel_name,
                metric_name=metric_name,
                metric_unit=metric_unit,
                metric_value=parsed_value,
            )
        )

    return records


def aggregate_metric_records(records: list[MetricRecord]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[float]] = {}
    units: dict[str, str] = {}
    for record in records:
        buckets.setdefault(record.metric_name, []).append(record.metric_value)
        if record.metric_name not in units:
            units[record.metric_name] = record.metric_unit

    output: dict[str, dict[str, Any]] = {}
    for name, values in buckets.items():
        values_sorted = sorted(values)
        count = len(values_sorted)
        median = values_sorted[count // 2] if count % 2 == 1 else (values_sorted[count // 2 - 1] + values_sorted[count // 2]) / 2.0
        output[name] = {
            "unit": units.get(name, ""),
            "count": count,
            "min": values_sorted[0],
            "max": values_sorted[-1],
            "median": median,
        }
    return output


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Compile, run, and profile CUDA binaries")
    parser.add_argument("--source", type=Path, help="Path to CUDA source (.cu)")
    parser.add_argument("--binary", type=Path, required=True, help="Path to output or existing binary")
    parser.add_argument("--run", action="store_true", help="Run binary after compile")
    parser.add_argument("--profile", action="store_true", help="Profile binary with ncu")
    parser.add_argument("--metrics", default="", help="Comma-separated NCU metrics")
    parser.add_argument("--timeout", type=int, default=240, help="Timeout in seconds for run/profile")
    parser.add_argument("args", nargs="*", help="Program arguments")
    args = parser.parse_args()

    if args.source is not None:
        compile_result = compile_cuda_source(args.source, args.binary, timeout_s=args.timeout)
        print("=== COMPILE ===")
        print(f"duration_s={compile_result.duration_seconds:.3f}")

    if args.run:
        run_result = run_binary(args.binary, args=args.args, timeout_s=args.timeout)
        print("=== RUN STDOUT ===")
        print(run_result.stdout.strip())
        if run_result.stderr.strip():
            print("=== RUN STDERR ===")
            print(run_result.stderr.strip())

    if args.profile:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] if args.metrics else None
        profile_result = profile_with_ncu(args.binary, args=args.args, metrics=metrics, timeout_s=args.timeout)
        records = parse_ncu_csv(profile_result.stdout)
        aggregate = aggregate_metric_records(records)
        print("=== NCU CSV ===")
        print(profile_result.stdout.strip())
        print("=== NCU AGGREGATE ===")
        for metric_name, stats in sorted(aggregate.items()):
            print(f"{metric_name}: {stats}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
