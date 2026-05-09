# Phase 2 Report

Student ID: 23302010025

## Agent Design

The submission runs `bash run.sh`, which launches `python3 -m agent.main`. The agent immediately writes a conservative, valid `optimized_lora.cu` so the evaluation root always contains a compilable implementation. It then generates several CUDA extension candidates for:

```text
Y = W X + A(B^T X), r = 16
```

Each candidate is compiled with the PyTorch extension toolchain, checked against an official-style PyTorch reference on synthetic tensors, benchmarked with CUDA events, and compared by geometric mean speedup over the configured public-size benchmark cases. Whenever a candidate becomes the current best, the agent atomically replaces `optimized_lora.cu`.

## Candidate Families

- ATen/cuBLAS fallback: `mm(W, X)`, `mm(B^T, X)`, and `addmm` accumulation.
- In-place addmm variants: accumulate the rank-16 term directly into `W @ X` with `addmm_` to avoid an extra output allocation.
- Direct cuBLAS variant: issue the three SGEMMs explicitly and use `beta=1` on the final skinny GEMM to update `Y` in place.
- Separate add variants: compute `A @ (B^T @ X)` as its own GEMM and add it to `W @ X`, avoiding `addmm(beta=1)` when that path is slower.
- Materialized `B^T` variants: explicitly make the rank-16 transposed panel contiguous before `B^T @ X` when that is faster than strided skinny GEMM.

## Reproducibility

The main knobs are exposed as environment variables:

```bash
LORA_AGENT_MAX_CANDIDATES=5
LORA_AGENT_BENCH_DIMS=3584,4096,4352,4608
LORA_AGENT_CORRECTNESS_DIMS=256
LORA_AGENT_WARMUP=4
LORA_AGENT_ITERS=10
```

The agent also writes `output.json` with candidate-level compile status, correctness errors, median runtimes, speedups, and the selected best candidate.

## Best Output ID

`d558fbf45dbe4dd0749e7dc0430e0bd7`

