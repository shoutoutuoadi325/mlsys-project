# Phase 3 Submission Report

## 1. Clear Runtime Organization

The submitted `run.sh` executes `agent.py`, and the agent deterministically writes the final inference runtime to `/workspace/engine.py`. The runtime exposes the required interface:

```python
def create_engine(model_config: dict, weight_dir: str, device: str = "cuda"):
    return Engine(model_config, weight_dir, device)
```

The generated `Engine` is organized around four responsibilities:

- Initialization and model loading: `__init__` selects the device and dtype, loads `model.pt`, reads all dimensions from `model_config`, prepares RoPE tables, and packs per-layer weights into a compact `self.layers` list.
- Request lifecycle API: `prefill`, `decode`, and `remove` implement the evaluator-facing interface. They preserve caller output order, validate request/token counts, and update or delete request state explicitly.
- Persistent request state and KV cache: `RequestState` stores each request's current sequence length plus its row inside a `CacheBlock`; `CacheBlock` stores per-layer key/value tensors with spare capacity.
- Transformer kernels and helpers: `_forward_prefill_batch`, `_decode_group`, `_decode_varlen_group`, `_rmsnorm`, RoPE helpers, attention helpers, and cache growth routines isolate the actual model computation from request scheduling.

This separation keeps the public API small while making the internal optimization decisions clear: batching happens in `prefill`/`decode`, storage decisions happen in `CacheBlock` and `_ensure_capacity`, and math decisions happen in the forward/decode helpers.

## 2. Benchmarking and Profiling for Decision Making

The optimization path was driven by repeated correctness checks and throughput submissions. The representative selected result is recorded in `/workspace/output_id3.txt`:

```text
e9ef5af9ac90250aa421d549ce868896
```

The selected public Submit3 result was:

| Case | tokens/s | decode tokens/s | peak memory MB |
|---|---:|---:|---:|
| 1 | 98495.3331 | 0.0000 | 1008.1279 |
| 2 | 8409.2248 | 934.3583 | 1063.9443 |
| 3 | 10512.3306 | 470.7014 | 1067.7271 |
| 4 | 20350.4098 | 466.0399 | 1630.3877 |

The main bottleneck identified from the benchmark pattern was decode, not model initialization. Initialization is outside the timed region, while every `decode()` call is timed and may happen many times per request. This led to the following decisions:

- KV cache instead of full recomputation: the baseline reran the whole prompt on each decode step. The final runtime stores per-layer K/V tensors during `prefill` and appends only the new token in `decode`, reducing decode work from full-sequence transformer passes to one-step attention over cached keys/values.
- Shared block cache for batched requests: requests created in the same prefill group share a `CacheBlock`, so common decode traces can update contiguous rows and slice a single backing tensor instead of repeatedly stacking unrelated cache tensors.
- Fused projections: Q/K/V weights and gate/up MLP weights are concatenated at initialization. Each layer then uses one QKV linear and one gate/up linear, reducing Python overhead and kernel launches in both prefill and decode.
- Prefill-only `torch.compile`: prefill batches have stable shapes after grouping by prompt length, so compilation can help. Decode shapes are more dynamic because request order and sequence lengths change across traces; decode compilation and a manual attention implementation were tested but rejected because they did not improve the official timing.
- Reusable variable-length decode scratch buffers: mixed traces may decode requests with different current lengths. `_decode_varlen_group` reuses preallocated K/V/mask buffers sized by geometric growth, avoiding repeated allocation churn in the timed path.

## 3. Iterative Improvement Process

The development process followed a baseline-to-optimized loop:

| Iteration | Observation | Change | Resulting Effect |
|---|---|---|---|
| Baseline | Public baseline was correct but decoded by rerunning the full sequence. | Implemented per-request KV cache. | Decode became incremental and avoided recomputing previous tokens. |
| Batched prefill | Prompt batches with equal lengths were common and shape-stable. | Grouped prefill requests by sequence length and ran `_forward_prefill_batch`. | Improved prefill throughput and made prefill suitable for `torch.compile`. |
| Decode grouping | Requests often advance at the same length after being prefilled together. | Added shared `CacheBlock`, contiguous-row fast path, and same-length `_decode_group`. | Reduced cache gathering overhead for common batched decode traces. |
| Mixed traces | Hidden traces can insert/remove requests and decode uneven-length batches. | Added request dictionary, `remove`, variable-length decode path, and reusable scratch buffers. | Preserved correctness and performance under non-uniform request patterns. |
| Kernel overhead | Per-layer Q/K/V and MLP projections were launch-heavy. | Fused QKV and gate/up projections during initialization. | Reduced timed compute overhead without changing reference math. |
| Candidate experiments | Decode compilation and manual decode attention did not improve submitted timing. | Kept PyTorch SDPA and limited compilation to prefill. | Avoided slower or less robust optimizations. |

The final result balances throughput and robustness: it keeps the code fully PyTorch-based for correctness stability while using cache layout, batching, fused projections, and selective compilation to target the measured bottlenecks.

## 4. Robust Handling of Model Configs and Request Patterns

The runtime does not hard-code model dimensions or paths. During `create_engine`, it reads the following values from `model_config`: `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `head_dim`, `hidden_size`, `vocab_size`, `rms_norm_eps`, `rope_theta`, `max_position_embeddings`, and `torch_dtype`. The engine derives `q_size`, `kv_size`, KV repeat factor, RoPE cache length, output vocabulary size, and tensor shapes from those values.

Weights are loaded from the evaluator-provided `weight_dir/model.pt`; layer tensors are discovered using the expected layer prefix pattern, but the number of layers comes from the config. This means the same generated `engine.py` can handle hidden model sizes as long as they follow the provided decoder-only architecture.

The request path is also dynamic:

- `prefill` accepts either a list of 1D tensors or tensor inputs, groups only by observed prompt length, and returns logits in the original request order.
- `decode` validates that every request has already been prefilled, accepts arbitrary request ordering, and chooses between same-length and variable-length decode paths.
- `remove` deletes request state without assuming requests finish in creation order.
- KV cache capacity grows geometrically through `_ensure_capacity`, so decode length is not fixed in advance.
- RoPE tables grow through `_ensure_rope` if hidden traces exceed the initially configured position cache.
- Empty batches return correctly shaped empty logits instead of crashing.

These choices are important for hidden evaluation because batch size, prompt length, decode steps, insertion/removal order, and request interleaving are not known ahead of time.

## 5. Reproducibility Through `run.sh` and Logs

The submission is reproducible through the provided `/run.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON:-python3}"

mkdir -p workspace
"$PYTHON_BIN" agent.py > workspace/results.log 2>&1
```

Running this script regenerates `/workspace/engine.py` from `agent.py` and records the agent execution log in `/workspace/results.log`. The agent also writes `/workspace/output3.md`, which documents the generation assumptions, config path, and weight path checked during generation.

The representative evidence ID is stored in:

```text
/workspace/output_id3.txt
```

with contents:

```text
e9ef5af9ac90250aa421d549ce868896
```

This gives the evaluator a concrete output ID tied to the selected optimized run, while `run.sh`, `agent.py`, `workspace/results.log`, `workspace/output3.md`, and `workspace/engine.py` together make the generated runtime and optimization evidence reproducible.

## 6. Final Summary

The final runtime is a dynamic PyTorch inference engine for decoder-only LLM evaluation. Its core performance features are KV caching, equal-length prefill batching, shared cache blocks, fused QKV and MLP projections, prefill-only compilation, and reusable variable-length decode buffers. Its robustness comes from deriving dimensions from `model_config.json`, loading weights from the evaluator-provided directory, preserving request order, supporting arbitrary decode/remove patterns, and growing caches as needed.
