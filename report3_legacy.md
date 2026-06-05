# Phase 3 Report: Automated LLM Inference Runtime

**Student ID:** 23302010025

---

## Overview

This phase requires building an *automated agent* that uses an LLM API to generate a complete, correct, and high-throughput `engine.py` for a decoder-only LLaMA-like inference runtime. The agent reads the public model configuration and weight schema, constructs a detailed prompt, calls the LLM, validates the result, and iterates with concrete feedback until the generated engine meets both correctness and throughput targets.

---

## Agent Design

### Entry Point

`run.sh` invokes `agent.py`, which orchestrates the full generation–validation–repair loop and writes the final `engine.py` and `output3.md` into `/workspace/`.

### Prompt Construction (`build_runtime_prompt`)

The agent dynamically builds a prompt that includes:

- **Model architecture specification**: exact interface (`create_engine`, `prefill`, `decode`, `remove`), LLaMA math (RMSNorm, RoPE even/odd convention, SwiGLU MLP, GQA repeat), and return shape requirements (`[batch_size, vocab_size]`).
- **Performance targets**: ≥ 45k tokens/s on prefill-heavy cases, ≥ 10k tokens/s on mixed cases, with explicit guidance on KV cache, batched prefill grouping, grouped decode, fused QKV/gate-up projections, cached RoPE tables, and `torch.inference_mode`.
- **Observed model config**: read live from `/target/model_config.json` so the prompt reflects the actual hidden evaluation config, not a hard-coded assumption.
- **Observed weight schema**: actual `state_dict` keys and shapes from `/target/weights/model.pt`, preventing the model from inventing non-existent keys (e.g., `rotary_emb.inv_freq`).
- **Common failure fixes**: proactive instructions to avoid known pitfalls (wrong logit shape, `torch.tensor` on a list of tensors, incorrect RoPE head-dim indexing, padding-induced last-token errors).
- **Previous candidate feedback**: if a prior attempt failed correctness or benchmarks, the exact error output or benchmark summary is appended so the model can do targeted repair.

### Generation Loop (`generate_candidate_with_feedback`)

The agent runs up to **6 attempts**. Each attempt:

1. Calls the LLM (temperature 0.15 on first attempt, 0.1 on retries).
2. Runs `auto_repair_source` to fix systematic issues the model tends to produce:
   - `config.key` → `config["key"]` for all known config fields.
   - `torch.tensor([ids for ids in list], dtype=torch.long)` → `torch.stack([ids.to(...) for ids in list])`.
3. Syntax-checks the source with `compile()`.
4. Runs the **public correctness evaluator** (`test_correctness.py`). If it fails, the full error log becomes the next-round feedback.
5. If correctness passes, runs the **public benchmark** (`benchmark_throughput.py`). The benchmark summary (tokens/s per case) is included in the feedback if throughput targets are not met.
6. Accepts immediately if correctness passes and case 1 ≥ 40k tokens/s and case 4 ≥ 9.5k tokens/s. Otherwise, retains the most recent correct candidate and continues trying.

After the loop, the best correct candidate is written to `/workspace/engine.py`.

### LLM Interface

The agent uses a plain `urllib.request` HTTP call (no external SDK dependency) to any OpenAI-compatible endpoint. The model and credentials are read from environment variables (`API_KEY`, `BASE_URL`, `PHASE3_MODEL`), making the agent fully compatible with the evaluation system's injected API key and model.

---

## Key Design Decisions

**No hard-coded dimensions.** Every architectural parameter (`hidden_size`, `num_heads`, `head_dim`, `intermediate_size`, etc.) is read from `model_config` at runtime. This is critical because the hidden evaluation uses a different model size than the public tiny-LLaMA config.

**Weight schema injection.** Loading the actual `state_dict` keys into the prompt eliminates a major failure mode: the LLM trying to use HuggingFace-style weight names that don't exist in the provided checkpoint.

**Correctness before throughput.** The loop gates on correctness first. A fast-but-wrong engine scores zero on correctness cases and zero on the throughput cases that depend on them.

**Concrete feedback over abstract hints.** Rather than saying "try to be faster", the feedback includes the exact benchmark numbers and a targeted checklist of the specific optimizations that were missing. This gives the repair model precise information to act on.

---

## Generated Engine Architecture

The engine produced by the agent in the best run implements:

- **Per-request KV cache** (`kv_cache: dict[int, list[(K, V)]]`): each request stores per-layer key and value tensors of shape `[num_kv_heads, seqlen, head_dim]`. Decode appends only the new token's K/V, avoiding full re-computation.
- **Batched prefill grouped by sequence length**: same-length requests are stacked into a single `[B, seqlen]` batch and processed in one forward pass with `is_causal=True` SDPA.
- **Batched decode grouped by cached length**: requests with the same KV length are batched; cached K/V tensors are stacked and concatenated with the new token's K/V before attention.
- **Fused QKV projection**: Q, K, V weight matrices concatenated into a single `[q_dim + 2*kv_dim, hidden]` matrix, halving the number of matmul launches per attention layer.
- **Fused gate/up projection**: gate and up weight matrices concatenated into `[2*intermediate, hidden]`, reducing MLP matmul launches from 2 to 1.
- **Precomputed RoPE tables**: cos/sin tables computed once at init and grown lazily. The even/odd rotation convention matches the reference exactly.
- **`F.scaled_dot_product_attention` on CUDA**: uses Flash Attention automatically for both prefill (causal) and decode (non-causal).
- **`@torch.inference_mode()`** on `prefill` and `decode`.

---

## Public Benchmark Results (Submission `04355a648a1278bf94c2dfab3eddbdbb`)

| Case | tokens/s | decode tokens/s | peak memory |
|------|----------|-----------------|-------------|
| 1 (prefill-heavy) | 46,023 | — | 1,042 MB |
| 2 | 5,979 | 664 | 1,138 MB |
| 3 | 4,788 | 214 | 1,003 MB |
| 4 (mixed) | 9,385 | 215 | 1,110 MB |

Correctness: **passed** (all cases, `atol=1e-2, rtol=1e-2`).

The generation succeeded on the **first attempt**: the LLM produced a correct and fast implementation in a single call, with no repair iterations needed.
