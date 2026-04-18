# MLSYS Course Project Report

## Student and Representative Output

Student ID: 23302010025
Representative output ID: 1b469567f94a0bdd5b147a369a6b3edc

This report summarizes the submitted GPU probing agent and the rationale for selecting the representative output ID.

## Agent Goal

The agent reads a target specification, probes requested hardware-related metrics using CUDA micro-benchmarks and Nsight Compute metrics, and writes one structured output file with numeric results and detailed evidence.

## System Design

### 1. Input and Output Contract

- Target spec input path defaults to /target/target_spec.json.
- Output path defaults to /workspace/output.json.
- Output includes numeric results, detailed per-target evidence, runtime summary, and a short reasoning section.

### 2. Planning Layer

- A target planner maps each target to one of three execution modes:
  - built-in probe
  - device attribute query
  - ncu metric query
- The planner can use an LLM plan when API variables are provided.
- If LLM planning is unavailable, deterministic fallback rules are used.

### 3. Probing Layer

- Built-in probes are used for targets like SM count and other hardware-intrinsic quantities.
- Device-attribute probes query CUDA device properties for metrics such as bus width and max frequencies.
- Nsight Compute probes collect runtime counters (for example DRAM read/write throughput and SOL percentages).

### 4. Reliability and Failure Handling

- The run writes a valid output file on both success and failure paths.
- Internal run state is persisted for debugging.
- If API credentials are absent, the reasoning component falls back to deterministic local summarization.

## Environment Interface

The implementation exposes model interfaces through environment variables:

- API_KEY
- BASE_MODEL
- BASE_URL

This is compatible with evaluation-side model injection.

## Representative Output Selection

I executed multiple submit-test runs and compared successful outputs by:

- completion and target success rate
- numeric completeness (non-null metrics)
- internal consistency of memory metrics
- overall run stability

I excluded runs that failed due server-side infrastructure issues (for example Docker image pull timeout), since those do not reflect agent quality.

The selected output ID is:

1b469567f94a0bdd5b147a369a6b3edc

## Result Snapshot of Selected Output

Run summary:

- target_count: 8
- success_count: 8
- failure_count: 0
- duration_seconds: 30.541779757011682

Key values:

- device__attribute_fb_bus_width: 384
- device__attribute_max_gpu_frequency_khz: 1695000
- device__attribute_max_mem_frequency_khz: 9751000
- launch__sm_count: 82
- gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed: 79.28
- sm__throughput.avg.pct_of_peak_sustained_elapsed: 10.085
- dram__bytes_read.sum.per_second: 363089349053.53503
- dram__bytes_write.sum.per_second: 362352599953.99

These values are coherent for a memory-dominant workload profile, with high memory throughput percentage and much lower SM throughput percentage.

## Limitations and Next Improvements

- Some ncu targets are currently measured with a single profiling trial; adding multi-trial median aggregation would improve robustness.
- Device attribute frequency may not always equal actual sustained boost frequency under lock/throttle settings; adding clock-under-load probes would improve fidelity.

## Final Submission Artifact

The required representative output ID is provided in output_id.txt.
