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
- ATen/cuBLAS plus fused rank-16 addition: the two matrix products use PyTorch/cuBLAS, while the low-rank accumulation is performed by custom CUDA kernels with several thread-block shapes.
- Direct cuBLAS variants: explicit SGEMM calls for `W@X` and `B^T@X`, followed by the same fused rank-16 addition kernel.

## Reproducibility

The main knobs are exposed as environment variables:

```bash
LORA_AGENT_MAX_CANDIDATES=6
LORA_AGENT_BENCH_DIMS=3584,4096
LORA_AGENT_CORRECTNESS_DIMS=256
LORA_AGENT_WARMUP=4
LORA_AGENT_ITERS=10
```

The agent also writes `output.json` with candidate-level compile status, correctness errors, median runtimes, speedups, and the selected best candidate.

## Best Output ID

TBD after running `/submit2`. Copy the best returned output id into `output_id2.txt` before the final report submission.
