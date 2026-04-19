# MLSYS Course Project Report

## Student and Representative Output

Student ID: 23302010025
Representative output ID: 64ee79425df7421281a7f110ac0cb310

This report summarizes the updated probing architecture and the rationale for selecting the representative output.

## Agent Goal

The agent reads target metrics from /target/target_spec.json, probes each target with an appropriate method (built-in CUDA probe, device attribute query, or Nsight Compute metric probe), and writes one structured output file to /workspace/output.json.

## Updated System Design

### 1. Entry and Orchestration

- run.sh launches python -m agent.main.
- The main loop loads the target spec, executes per-target probing, and emits one JSON artifact with:
  - results (numeric summary)
  - details (method, confidence, evidence, errors)
  - summary (runtime and success/failure counts)
  - reasoning (LLM-generated or deterministic fallback)

### 2. Planning Layer

- TargetPlanner maps each target to one of three plan kinds:
  - builtin_probe
  - device_attribute
  - ncu_metric
- Planner behavior:
  - Uses LLM planning when API_KEY and BASE_MODEL are available.
  - Falls back to deterministic rules when LLM is unavailable or plan validation fails.

### 3. Probing Layer (Architecture Change)

- Built-in probes and device attribute probes remain deterministic and compiled locally.
- ncu_metric probing now supports two execution paths:
  - Preferred path: LLM-generated CUDA benchmark source, compiled and profiled with ncu.
  - Fallback path: pre-defined static metric workload benchmark.
- LLM benchmark generation details:
  - Controlled by LLM_BENCHMARK_ENABLED (default enabled when model/client are available).
  - Uses prompt-based code generation and compile-retry with error feedback.
  - Records benchmark_source, benchmark_path, and generation attempts in evidence.
- If LLM benchmark generation fails, the system automatically falls back and records llm_benchmark_fallback and llm_benchmark_error.

### 4. Reliability and Failure Handling

- Output is always written in both success and failure paths.
- Run state and last_error are persisted for debugging.
- Reasoning generation also has a deterministic fallback when API access is unavailable.

## Environment Interface

The implementation is compatible with evaluation-side model injection through:

- API_KEY
- BASE_MODEL
- BASE_URL

Additional runtime controls include:

- LLM_BENCHMARK_ENABLED
- TARGET_SPEC_PATH
- OUTPUT_PATH
- STATE_PATH
- GENERATED_DIR
- PROMPT_DIR
- COMPILE_TIMEOUT_S, RUN_TIMEOUT_S, PROFILE_TIMEOUT_S, MAX_TRIALS

## Representative Output Quality

Selected output file: 64ee79425df7421281a7f110ac0cb310

Run summary:

- target_count: 8
- success_count: 8
- failure_count: 0
- duration_seconds: 687.5979469179874

Key probed values:

- device__attribute_fb_bus_width: 384 bits
- device__attribute_max_gpu_frequency_khz: 1695000 kHz
- device__attribute_max_mem_frequency_khz: 9751000 kHz
- launch__sm_count: 82
- dram__bytes_read.sum.per_second: 850805207349.38 byte/second
- dram__bytes_write.sum.per_second: 365250907259.41003 byte/second
- gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed: 88.63 %
- sm__throughput.avg.pct_of_peak_sustained_elapsed: 50.82 %

Validation notes:

- All targets have status = ok.
- No result values are null, negative, or exactly zero.
- Planning source is LLM for all 8 targets.
- LLM-generated benchmark path was used for all 4 ncu metrics.
- No llm_benchmark_fallback occurred in the selected run.

## Compliance with README Requirements

- /workspace/run.sh exists and runs the agent.
- Agent reads /target/target_spec.json by default.
- Agent writes a single output artifact under /workspace/output.json.
- Model API interface is exposed through API_KEY, BASE_MODEL, BASE_URL.
- Representative output ID is provided in output_id.txt.

## Final Submission Artifact

The representative output ID is recorded in output_id.txt.
