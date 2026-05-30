import argparse
import importlib.util
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class BenchResult:
    case_name: str
    elapsed_ms: float
    prefill_tokens: int
    decode_tokens: int
    total_tokens: int
    tokens_per_second: float
    decode_tokens_per_second: float
    peak_memory_mb: float


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def load_student_engine(engine_path):
    spec = importlib.util.spec_from_file_location("student_engine", engine_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timed_run(engine, events, device):
    prefill_tokens = 0
    decode_tokens = 0

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    sync(device)
    start = time.perf_counter()

    with torch.no_grad():
        for event in events:
            op = event["op"]

            if op == "prefill":
                request_ids = event["request_ids"]
                input_ids = event["input_ids"]
                engine.prefill(request_ids, input_ids)
                prefill_tokens += sum(int(x.numel()) for x in input_ids)

            elif op == "decode":
                request_ids = event["request_ids"]
                token_ids = event["token_ids"]
                engine.decode(request_ids, token_ids)
                decode_tokens += int(token_ids.numel())

            elif op == "remove":
                engine.remove(event["request_ids"])

            else:
                raise ValueError(f"unknown op: {op}")

    sync(device)
    end = time.perf_counter()

    elapsed_ms = (end - start) * 1000.0
    total_tokens = prefill_tokens + decode_tokens

    peak_memory_mb = 0.0
    if device.startswith("cuda"):
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return elapsed_ms, prefill_tokens, decode_tokens, total_tokens, peak_memory_mb


def make_prefill_case(batch_size, prompt_len, vocab_size, device):
    request_ids = list(range(batch_size))
    input_ids = [
        torch.randint(0, vocab_size, (prompt_len,), dtype=torch.long, device=device)
        for _ in range(batch_size)
    ]

    return [
        {"op": "prefill", "request_ids": request_ids, "input_ids": input_ids},
        {"op": "remove", "request_ids": request_ids},
    ]


def make_decode_case(batch_size, prompt_len, decode_steps, vocab_size, device):
    request_ids = list(range(batch_size))
    input_ids = [
        torch.randint(0, vocab_size, (prompt_len,), dtype=torch.long, device=device)
        for _ in range(batch_size)
    ]

    events = [{"op": "prefill", "request_ids": request_ids, "input_ids": input_ids}]

    for _ in range(decode_steps):
        token_ids = torch.randint(
            0,
            vocab_size,
            (batch_size,),
            dtype=torch.long,
            device=device,
        )
        events.append(
            {"op": "decode", "request_ids": request_ids, "token_ids": token_ids}
        )

    events.append({"op": "remove", "request_ids": request_ids})
    return events


def make_mixed_case(vocab_size, device):
    events = []
    active = set()
    next_request_id = 0

    schedule = [
        ("prefill", 4, 64),
        ("decode", 4, None),
        ("decode", 4, None),
        ("prefill", 2, 128),
        ("decode", 6, None),
        ("remove", 2, None),
        ("prefill", 4, 32),
        ("decode", 8, None),
        ("decode", 8, None),
        ("remove", 8, None),
    ]

    for op, count, prompt_len in schedule:
        if op == "prefill":
            request_ids = list(range(next_request_id, next_request_id + count))
            next_request_id += count
            active.update(request_ids)

            input_ids = [
                torch.randint(
                    0,
                    vocab_size,
                    (prompt_len,),
                    dtype=torch.long,
                    device=device,
                )
                for _ in request_ids
            ]
            events.append(
                {"op": "prefill", "request_ids": request_ids, "input_ids": input_ids}
            )

        elif op == "decode":
            request_ids = sorted(active)[:count]
            token_ids = torch.randint(
                0,
                vocab_size,
                (len(request_ids),),
                dtype=torch.long,
                device=device,
            )
            events.append(
                {"op": "decode", "request_ids": request_ids, "token_ids": token_ids}
            )

        elif op == "remove":
            request_ids = sorted(active)[:count]
            for rid in request_ids:
                active.remove(rid)
            events.append({"op": "remove", "request_ids": request_ids})

    if active:
        events.append({"op": "remove", "request_ids": sorted(active)})

    return events


def benchmark_case(
    case_name,
    engine_mod,
    model_config,
    weight_dir,
    events,
    device,
    warmup,
    repeat,
):
    warmup_engine = engine_mod.create_engine(model_config, weight_dir, device)
    for _ in range(warmup):
        timed_run(warmup_engine, events, device)

    measurements = []
    for _ in range(repeat):
        engine = engine_mod.create_engine(model_config, weight_dir, device)
        measurements.append(timed_run(engine, events, device))

    measurements.sort(key=lambda x: x[0])
    elapsed_ms, prefill_tokens, decode_tokens, total_tokens, peak_memory_mb = measurements[
        len(measurements) // 2
    ]
    elapsed_s = elapsed_ms / 1000.0

    return BenchResult(
        case_name=case_name,
        elapsed_ms=elapsed_ms,
        prefill_tokens=prefill_tokens,
        decode_tokens=decode_tokens,
        total_tokens=total_tokens,
        tokens_per_second=total_tokens / elapsed_s,
        decode_tokens_per_second=decode_tokens / elapsed_s if decode_tokens else 0.0,
        peak_memory_mb=peak_memory_mb,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weight-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(0)

    with Path(args.model_config).open() as f:
        model_config = json.load(f)

    vocab_size = int(model_config["vocab_size"])
    engine_mod = load_student_engine(args.engine)

    cases = {
        "prefill": make_prefill_case(
            batch_size=4,
            prompt_len=128,
            vocab_size=vocab_size,
            device=device,
        ),
        "decode": make_decode_case(
            batch_size=8,
            prompt_len=32,
            decode_steps=16,
            vocab_size=vocab_size,
            device=device,
        ),
        "mixed": make_mixed_case(vocab_size=vocab_size, device=device),
    }

    results = []
    for case_name, events in cases.items():
        result = benchmark_case(
            case_name=case_name,
            engine_mod=engine_mod,
            model_config=model_config,
            weight_dir=args.weight_dir,
            events=events,
            device=device,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        results.append(asdict(result))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

