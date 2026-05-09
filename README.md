# MLSYS Course Project Phase 2

This workspace contains a Phase 2 optimization agent for the LoRA operator:

```text
Y = W X + A(B^T X), r = 16
```

The submission entry point is:

```bash
bash run.sh
```

During execution the agent keeps a valid `optimized_lora.cu` in the submission root. It starts from a conservative ATen/cuBLAS fallback, then compiles and benchmarks generated CUDA extension candidates when CUDA and `nvcc` are available.

Useful environment overrides:

```bash
LORA_AGENT_MAX_CANDIDATES=6
LORA_AGENT_BENCH_DIMS=3584,4096
LORA_AGENT_CORRECTNESS_DIMS=256
LORA_AGENT_WARMUP=4
LORA_AGENT_ITERS=10
```

After a `/submit2` run finishes, put the best returned output id into `output_id2.txt` for the final report submission.
