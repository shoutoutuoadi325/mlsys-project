import argparse
import importlib.util
import json
from pathlib import Path

import torch

from reference_model import ReferenceModel


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


def assert_close(name, student_logits, reference_logits, atol, rtol):
    student = student_logits.detach().float().cpu()
    reference = reference_logits.detach().float().cpu()

    if student.shape != reference.shape:
        raise AssertionError(
            f"{name}: shape mismatch, student={tuple(student.shape)}, "
            f"reference={tuple(reference.shape)}"
        )

    if not torch.allclose(student, reference, atol=atol, rtol=rtol):
        diff = (student - reference).abs()
        max_abs = float(diff.max())
        ref_scale = reference.abs().clamp_min(1e-12)
        max_rel = float((diff / ref_scale).max())
        raise AssertionError(
            f"{name}: logits mismatch, max_abs={max_abs:.6g}, max_rel={max_rel:.6g}"
        )


def reference_last_logits(ref_model, ids):
    return ref_model.forward(ids.unsqueeze(0))[0, -1, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--weight-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(7)

    with Path(args.model_config).open() as f:
        model_config = json.load(f)

    vocab_size = int(model_config["vocab_size"])
    engine_mod = load_student_engine(args.engine)
    engine = engine_mod.create_engine(model_config, args.weight_dir, device)
    ref_model = ReferenceModel(model_config, args.weight_dir, device)

    request_tokens = {}

    def run_prefill(case_name, request_ids, input_ids):
        student_logits = engine.prefill(request_ids, input_ids)
        expected = []
        for rid, ids in zip(request_ids, input_ids):
            rid = int(rid)
            ids = ids.to(device=device, dtype=torch.long)
            request_tokens[rid] = ids.clone()
            expected.append(reference_last_logits(ref_model, ids))
        expected_logits = torch.stack(expected, dim=0)
        assert_close(case_name, student_logits, expected_logits, args.atol, args.rtol)

    def run_decode(case_name, request_ids, token_ids):
        token_ids = token_ids.to(device=device, dtype=torch.long)
        student_logits = engine.decode(request_ids, token_ids)
        expected = []
        for rid, token in zip(request_ids, token_ids):
            rid = int(rid)
            request_tokens[rid] = torch.cat([request_tokens[rid], token.reshape(1)])
            expected.append(reference_last_logits(ref_model, request_tokens[rid]))
        expected_logits = torch.stack(expected, dim=0)
        assert_close(case_name, student_logits, expected_logits, args.atol, args.rtol)

    def run_remove(request_ids):
        engine.remove(request_ids)
        for rid in request_ids:
            request_tokens.pop(int(rid), None)

    with torch.no_grad():
        ids0 = torch.randint(0, vocab_size, (11,), device=device)
        run_prefill("single_prefill", [0], [ids0])

        tok0 = torch.randint(0, vocab_size, (1,), device=device)
        run_decode("single_decode", [0], tok0)

        ids1 = torch.randint(0, vocab_size, (7,), device=device)
        ids2 = torch.randint(0, vocab_size, (13,), device=device)
        run_prefill("multi_prefill", [1, 2], [ids1, ids2])

        toks = torch.randint(0, vocab_size, (3,), device=device)
        run_decode("multi_decode", [0, 1, 2], toks)

        run_remove([1])

        ids3 = torch.randint(0, vocab_size, (5,), device=device)
        run_prefill("insert_after_remove", [3], [ids3])

        toks = torch.randint(0, vocab_size, (3,), device=device)
        run_decode("decode_after_remove", [0, 2, 3], toks)

        run_remove([0, 2, 3])

    print(
        json.dumps(
            {
                "status": "passed",
                "engine": args.engine,
                "model_config": args.model_config,
                "device": device,
                "atol": args.atol,
                "rtol": args.rtol,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

