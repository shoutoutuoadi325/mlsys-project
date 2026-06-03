# Phase 3 Submission Report

## Selected Output ID

1106068f11ebfdfe4beb0a0adcc73ee6

## Result File

1106068f11ebfdfe4beb0a0adcc73ee63.md

## Summary

The submitted agent generates `workspace/engine.py` through `run.sh`. The generated runtime implements `create_engine(model_config, weight_dir, device)` and supports `prefill`, `decode`, and `remove` with per-request state and KV cache management.

The selected version uses shared KV cache blocks for batched requests, fused QKV and gate/up projections, decode RoPE lookup without GPU-to-CPU synchronization, and prefill-only `torch.compile`. Decode compilation was tested and rejected because it hurt the official case 2 timing.

## Latest Submit3 Result

| Case | tokens/s | decode tokens/s | peak memory MB |
|---|---:|---:|---:|
| 1 | 96591.3987 | 0.0000 | 538.1797 |
| 2 | 8151.6861 | 905.7429 | 828.7827 |
| 3 | 6843.6692 | 306.4330 | 759.9634 |
| 4 | 13340.8618 | 305.5159 | 1296.4146 |

Approximate public `outputs3` ranking at the time of measurement: 40 / 207 by four-case geometric mean.

## Notes

The latest reasoning output confirms that the agent reads the hidden evaluation inputs from `/target/model_config.json` and `/target/weights/model.pt`.
