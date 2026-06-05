# MLSYS Course Project Final Report

Student ID: 23302010025

本项目分为 Phase 1、Phase 2 和 Phase 3 三个阶段，整体目标是从 GPU 性能认知出发，逐步构建能够自动分析、自动优化、自动生成系统代码的 agent。三个阶段的工作分别覆盖硬件性能探测、CUDA 算子优化，以及 LLM 推理 runtime 自动生成，体现了从 profiling 到 operator tuning 再到 inference system generation 的递进关系。

## Phase 1: GPU 性能分析与硬件探测

Phase 1 的任务是构建一个能够读取目标指标、自动选择探测方法，并输出结构化结果的 GPU profiling agent。实现中，`run.sh` 启动 `python -m agent.main`，主流程读取 `/target/target_spec.json`，再由规划层为每个指标选择 `builtin_probe`、`device_attribute` 或 `ncu_metric` 三类执行路径。

该阶段的核心设计是把硬件探测抽象成可组合的 probe。对于 SM 数量、核心频率、缓存/显存延迟、带宽等指标，agent 使用本地 CUDA microbenchmark 进行确定性测量；对于设备属性，直接通过 CUDA runtime 查询；对于 Nsight Compute 指标，则支持静态 benchmark 和 LLM 生成 benchmark 两条路径。LLM 可根据目标 metric 生成 CUDA 源码，系统再进行编译、运行和 `ncu` profiling；如果生成或编译失败，则自动回退到内置 benchmark。

代表性提交结果中，8 个目标指标全部成功，包含显存总线宽度、GPU/显存频率、SM 数量、DRAM 读写吞吐、compute-memory throughput 和 SM throughput 等。Phase 1 的主要收获是建立了一个可靠的自动 profiling 框架：它不仅能测量数值，还能记录方法、置信度、trial 统计、错误信息和推理说明，为后续自动优化提供了硬件层面的反馈基础。代表输出 ID 为 `64ee79425df7421281a7f110ac0cb310`。

## Phase 2: LoRA 算子自动优化

Phase 2 的任务是为 LoRA-style 算子

```text
Y = W X + A(B^T X), r = 16
```

构建优化 agent。与手写单一 kernel 不同，本阶段要求 agent 在运行期间维护一个始终可编译的 `optimized_lora.cu`，并在 30 分钟时间预算内自动生成候选实现、编译、校验、benchmark、比较并替换当前最优版本。

实现中，agent 首先写入保守的 ATen/cuBLAS fallback，保证任意时刻评测系统都能读取到合法实现。随后它在公开尺寸范围内生成 synthetic tensors，对多个候选进行 PyTorch reference correctness check 和 CUDA event benchmark。候选族包括基础 `mm + addmm`、in-place `addmm_`、显式 materialized `B^T`、直接 cuBLAS 三次 SGEMM，以及其他低秩项累加思路。评分采用各 benchmark 维度 speedup 的几何平均值，当前最优候选会原子替换到 `optimized_lora.cu`。

最终报告记录的最佳实现是 `aten_addmm_inplace_strided_bt`，即先计算 `W @ X` 和 `B^T @ X`，再用 `Y.addmm_(A, T, 1.0, 1.0)` 原地累加低秩项，减少一次额外输出分配，同时保持与官方 PyTorch reference 接近的数值行为。Phase 2 的主要收获是把 Phase 1 的“观测与判断”推进为“生成与选择”：agent 不只分析性能，而是直接参与 CUDA 实现空间的搜索。最佳输出 ID 为 `d558fbf45dbe4dd0749e7dc0430e0bd7`。

## Phase 3: 自动生成 LLM 推理 Runtime

Phase 3 的任务是构建一个 agent，用 LLM API 自动生成 `workspace/engine.py`，实现 decoder-only LLaMA-like runtime。生成的 runtime 需要从 `model_config.json` 和 `model.pt` 动态构造模型，不硬编码隐藏尺寸，并支持 `create_engine`、`prefill`、`decode` 和 `remove` 接口。正确性通过与官方 reference logits 的 `torch.allclose(atol=1e-2, rtol=1e-2)` 比较验证，性能则通过 prefill、decode 和 mixed serving trace 测量。

本阶段 agent 的重点在 prompt construction 和 feedback loop。它会读取公开 config 和实际权重 schema，把模型结构、权重 key、RMSNorm、RoPE、GQA、SwiGLU、KV cache、返回 logits shape 等要求写入提示词，并把常见错误提前列入修复指令。生成代码后，agent 会进行语法检查、自动修复若干常见模式、运行 public correctness evaluator，再运行 throughput benchmark；若失败，则把具体错误或 benchmark 数字作为下一轮反馈，最多尝试 6 次。

最佳生成 runtime 使用 per-request KV cache，prefill 按相同序列长度分组批处理，decode 按缓存长度分组批处理，并结合 fused QKV projection、fused gate/up projection、缓存 RoPE table、`torch.inference_mode()` 和 CUDA 上的 `scaled_dot_product_attention` 来减少重复计算和 kernel launch。公开 benchmark 中 correctness 全部通过，prefill-heavy case 达到约 46k tokens/s，mixed case 达到约 9.4k tokens/s。Phase 3 的主要收获是将 agent 能力从单个算子扩展到完整推理系统：agent 需要同时维护接口语义、模型数学正确性、请求状态管理和吞吐优化。提交 ID 为 `04355a648a1278bf94c2dfab3eddbdbb3`。

## Overall Summary

三个阶段共同体现了 MLSYS 项目的主线：用 agent 自动化传统上依赖专家经验的系统工程工作。Phase 1 解决“如何理解硬件和性能瓶颈”，Phase 2 解决“如何围绕目标算子自动搜索更快实现”，Phase 3 则解决“如何根据规格自动生成正确且高吞吐的推理 runtime”。项目最终形成了一条从 profiling、benchmarking、candidate search 到 runtime generation 的完整闭环。

从实现经验看，正确性始终是优化的前提：无论是硬件指标探测、LoRA 输出对齐，还是 LLM logits 对齐，都必须先建立可验证的 reference 和稳定的错误反馈。其次，agent 的可靠性依赖明确的接口契约、保守 fallback、结构化日志和可复现实验配置。最后，LLM 在系统优化中的价值并不只是“生成代码”，更在于和编译器、profiler、benchmark、validator 组成反馈循环，从而把开放式探索转化为可评估、可迭代的工程过程。
