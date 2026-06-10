# MLSys Course Project Final Report

## 1. Basic Information

- Name: 张之麒
- Student ID: 23302010025
- Report date: 2026-06-10

## 2. Project Overview

这个项目的三个阶段分别对应 GPU 性能画像、LoRA 算子优化和 LLM 推理 runtime 生成。我的整体思路不是只手写一个固定答案，而是把每个阶段都做成一个可以根据目标、配置或评测反馈自动调整的 agent：Phase 1 根据 `target_spec.json` 规划探针并生成/运行 CUDA microbenchmark；Phase 2 生成多个 LoRA CUDA extension 候选、编译、校验、benchmark 后选择最优版本；Phase 3 根据 `model_config.json` 和 `model.pt` 的真实权重 schema 生成 `workspace/engine.py`，并用 public correctness/throughput evaluator 做闭环修复。

全自动部分主要包括：读取评测输入、构造 prompt、生成代码、编译/运行、解析指标、保存输出文件，以及在 Phase 2/3 中用正确性和性能结果筛选候选。仍然需要人工介入的部分包括：选择最终提交的 output id、解释服务端 hidden 结果、以及在部分服务端结果没有完整保存到仓库时手动整理证据。整个项目中 agent 工作流的变化很明显：Phase 1 更偏“测量任务编排”，Phase 2 变成“候选搜索 + 严格 correctness gate”，Phase 3 则进一步变成“代码生成 + evaluator 反馈修复”。

## 3. Phase 1: GPU Profiling Agent

Phase 1 的目标是让 agent 自动识别并测量 GPU 硬件/性能指标。入口是 `run.sh` 调用 `python -m agent.main`，agent 读取 `/target/target_spec.json`，对每个 target 生成一个 `TargetPlan`，再由 `ProbeExecutor` 执行。规划层支持三类动作：

- `builtin_probe`：例如 `launch__sm_count` 映射到 `physical_sm_count`，通过 SMID discovery kernel 统计物理 SM 数。
- `device_attribute`：例如 `device__attribute_fb_bus_width`、`device__attribute_max_gpu_frequency_khz`，通过 `cudaGetDeviceProperties` 读取。
- `ncu_metric`：例如 `dram__bytes_read.sum.per_second`，优先由 LLM 生成 CUDA benchmark，再用 Nsight Compute 采集指定 metric；失败时回退到静态 benchmark。

代表性输出为 `64ee79425df7421281a7f110ac0cb310`，8 个 target 全部成功，运行时间约 687.60 秒。

| Metric | Method | Value | Evidence |
| --- | --- | ---: | --- |
| `device__attribute_fb_bus_width` | device attribute | 384 bits | 3 次读取完全一致 |
| `device__attribute_max_gpu_frequency_khz` | device attribute | 1,695,000 kHz | 3 次读取完全一致 |
| `device__attribute_max_mem_frequency_khz` | device attribute | 9,751,000 kHz | 3 次读取完全一致 |
| `launch__sm_count` | SMID discovery kernel | 82 | 3 次 kernel 输出一致 |
| `dram__bytes_read.sum.per_second` | LLM-generated NCU benchmark | 850.81 GB/s | NCU CSV 聚合 |
| `dram__bytes_write.sum.per_second` | LLM-generated NCU benchmark | 365.25 GB/s | NCU CSV 聚合 |
| `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | LLM-generated NCU benchmark | 88.63% | NCU metric |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | LLM-generated NCU benchmark | 50.82% | NCU metric |

一个典型 profiling 命令来自输出 evidence：

```text
/workspace/.generated/bin/probe_device_attribute fb_bus_width
ncu -f --target-processes all --metrics dram__bytes_read.sum.per_second /workspace/.generated/bin/probe_metric_llm_...
```

我遇到的一个误导性测量是早期输出 `d55d8a96cf521018ee69b683c6b0f8ca`。它同样 8/8 成功，但 `gpu__compute_memory_throughput` 只有 13.595%，`sm__throughput` 却达到 82.22%。这说明“命令成功”和“测量到目标瓶颈”不是一回事：早期 LLM benchmark 更像混合计算 workload，不能稳定代表 memory-throughput pressure。后续我把 evidence 中的 `benchmark_source`、`record_count`、`aggregate` 都保留下来，并选择更贴近目标 metric 的 LLM-generated microbenchmark 作为最终代表输出。

## 4. Phase 2: LoRA Operator Optimization Agent

Phase 2 优化的算子是：

```text
Y = W X + A(B^T X), r = 16
```

agent 的入口是 `python3 -m agent.main`。它启动后首先写入一个保守、可编译的 `optimized_lora.cu`，避免后续候选生成/编译失败导致根目录没有有效提交文件。之后它枚举 `candidate_suite()` 中的候选，用 PyTorch extension toolchain 编译，先在 synthetic tensor 上和 PyTorch reference 比 correctness，再用 CUDA event 记录 median runtime，并以多个维度上的几何平均 speedup 选择当前 best。

| Version | Main idea | Correct? | Runtime / speedup | What I learned |
| --- | --- | --- | --- | --- |
| v1 `aten_addmm_fallback` | `Y=mm(W,X)`，`T=mm(B^T,X)`，再 `addmm` | Yes | agent 内部几何平均约 1.00077x；早期服务端 case: 2.0572x / 1.9080x / 0.9624x | cuBLAS/ATen baseline 很强，简单稳定版本很有竞争力 |
| v2 `aten_addmm_inplace_strided_bt` | 用 `Y.addmm_` 原地累加低秩项 | Yes, guarded by reference check | 完整候选日志未保留，未成为最终 best | 少一次输出分配有帮助，但 strided `B^T` 不一定比 ATen fallback 稳定 |
| v3 `cublas_three_sgemm_inplace` | 直接调用三次 SGEMM，最后一次 `beta=1` 写回 `Y` | Correctness risk, guarded | 未成为最终 best | 手写 cuBLAS 要特别小心 row-major/column-major 解释和 leading dimension |
| v4 `aten_addmm_inplace_contiguous_bt` | 显式 materialize `B^T.contiguous()` | Yes, guarded | 未成为最终 best | 对 skinny GEMM 可能更友好，但 materialization 也会带来额外成本 |
| final submitted | 保守 ATen/cuBLAS + in-place accumulation | Yes | output id `d558fbf45dbe4dd0749e7dc0430e0bd7`; hidden speedups in saved report: 2.3263x / 1.9596x / 0.9807x | 最终收益主要来自选择稳定 cuBLAS path，而不是复杂自定义 kernel |

最重要的瓶颈是两个大矩阵乘法主导总时间，而 rank-16 LoRA 项虽然数学上低秩，实际表达为 `B^T @ X` 和 `A @ T` 后仍然会触发较大的 GEMM/内存流量。一个看起来有吸引力但没有成为最终方案的方向是先构造 `W + A @ B^T`，再做一次大 GEMM；它减少了公式层面的乘法次数，但会 materialize 一个 `[d,d]` 临时矩阵，破坏了低秩项的内存优势。

一个 correctness 风险来自手写 cuBLAS 版本。PyTorch 张量是 row-major，而 cuBLAS API 默认按 column-major 解释，所以 `W @ X` 在代码中实际要按 `Y^T = X^T @ W^T` 的方式调用。这个问题的教训是：Phase 2 不能让 benchmark 先行，必须先计算 `max_abs` 和 `rel_l2`，否则一个很快的转置错误会被误认为优化成功。agent 中最终采用的阈值是 `rel_l2 <= 1e-5` 且 `max_abs <= 1e-2`。

## 5. Phase 3: Inference Runtime Agent

Phase 3 的 agent 位于 `phase3_submission/agent.py`，目标是生成一个完整的 `workspace/engine.py`，提供：

```python
create_engine(model_config, weight_dir, device="cuda")
prefill(request_ids, input_ids)
decode(request_ids, token_ids)
remove(request_ids)
```

生成过程不是盲目让 LLM 写代码。agent 会先读取 `target/model_config.json`，再尝试加载 `target/weights/model.pt`，把真实 state_dict keys 和 shapes 写进 prompt。这样可以避免模型发明 HuggingFace 风格的不存在权重，例如 `rotary_emb.inv_freq`。每一轮生成后，agent 会做 `compile()` syntax check，然后运行 public correctness evaluator；通过后再运行 throughput benchmark。如果正确性失败，下一轮 prompt 会带上完整错误日志；如果正确但不够快，下一轮 prompt 会带上 case-level tokens/s、decode tokens/s 和优化 checklist。

最终报告中的 best run 使用 output id `04355a648a1278bf94c2dfab3eddbdbb3`，正确性通过，public benchmark 如下：

| Case | tokens/s | decode tokens/s | peak memory |
| --- | ---: | ---: | ---: |
| 1, prefill-heavy | 46,023 | 0 | 1,042 MB |
| 2 | 5,979 | 664 | 1,138 MB |
| 3 | 4,788 | 214 | 1,003 MB |
| 4, mixed | 9,385 | 215 | 1,110 MB |

生成出的 runtime 设计包含这些关键点：

- 动态读取 `model_config` 中的 layer/head/hidden/intermediate/vocab/rope 参数，不硬编码 public tiny config。
- 加载 `weight_dir/model.pt`，用 plain Python containers 保存权重，避免 `nn.Module.load_state_dict(strict=True)` 引入额外 key。
- `prefill(...)` 按相同 prompt length 分组 batch，同组直接 `torch.stack`，返回 `[batch, vocab]` 的 last-token logits。
- `decode(...)` 对已有 request 追加一个 token，按当前 KV length 分组 batch，返回新 token 的 logits。
- `remove(...)` 只删除指定 request id 的状态，防止影响其他并发请求。
- 实现 per-request KV cache，每个 request 保存每层 K/V，decode 只计算新 token 并与 cache 拼接。
- 使用 fused QKV、fused gate/up、cached RoPE table、`torch.inference_mode()` 和 `F.scaled_dot_product_attention` 减少 Python/Kernel overhead。

一个 request-state 相关 bug 是早期候选容易把 `input_ids` 这个 tensor list 写成 `torch.tensor(list_of_tensors)`。这会在 PyTorch 中触发类似“only integer tensors of a single element can be converted to an index”的错误，也会破坏不同长度请求的处理。最终 agent 在 `auto_repair_source` 中加入了正则修复，把这类写法替换成：

```python
torch.stack([ids.to(device=device, dtype=torch.long) for ids in group], dim=0)
```

另一个 hidden-case 设计决策是禁止硬编码维度。Phase 3 历史提交里有一次“修复硬编码问题但是结果不高”，这说明只适配 public config 可以快速通过本地样例，却不能可靠适配 hidden model。最终 prompt 明确要求所有维度来自 `model_config`，权重名来自实际 `state_dict` schema。

## 6. Cross-Phase Reflection

三个阶段让我对 MLSys 的理解从“写能跑的代码”变成了“构造能解释、能验证、能迭代的系统”。Phase 1 的核心教训是 profiling 不是简单读一个指标；必须知道 workload 是否真的激活了目标硬件路径。一个 Nsight metric 数字本身不够，采样次数、benchmark source、profile duration 和不同输出之间的差异都要保存下来。

Phase 2 让我更直观地看到库函数 baseline 的强度。直觉上 rank-16 LoRA 很适合手写 kernel，但实际主耗时常常仍在大 GEMM 上，cuBLAS 已经做了大量底层优化。LLM agent 最适合帮助我快速生成候选和封装评测循环，但判断哪些优化值得保留，仍然需要理解内存布局、temporary allocation 和 numerical tolerance。

Phase 3 则把系统问题暴露得更完整：推理 runtime 不只是 transformer math，还包括请求生命周期、cache 状态、batching 策略、hidden config 泛化、输出 shape、dtype 和 evaluator contract。相比 Phase 1/2，Phase 3 最难自动化的是“保持正确性同时提吞吐”，因为一个小的 RoPE convention 或 last-token indexing 错误就会让所有 benchmark 失去意义。

agent 工作流也在不断改进：Phase 1 中我开始保存 structured evidence；Phase 2 中加入 compile/correctness/benchmark gate；Phase 3 中进一步把 evaluator 的失败日志送回 LLM 做修复。最容易让 LLM 帮忙的是生成样板代码、候选实现和 prompt checklist；最需要人工系统判断的是判断 measurement 是否可信、哪些优化只是局部漂亮但整体不划算、以及 hidden case 可能从哪里打破 public assumption。

## 7. Mistakes, Pitfalls, and Debugging Stories

| Problem | Symptom | Wrong hypothesis | Actual cause | Fix | Lesson |
| --- | --- | --- | --- | --- | --- |
| Phase 1 memory metric 不可信 | 早期输出里 memory throughput 13.595%，但 SM throughput 82.22% | GPU 本身 memory 利用率低 | microbenchmark 没有稳定压到目标 memory path，metric 成功不等于 workload 合适 | 选择更贴近 metric 的 LLM-generated benchmark，并保存 `aggregate`/`record_count` | profiling 要验证 workload，不只验证命令返回码 |
| Phase 2 手写 cuBLAS 方向风险高 | 候选可能很快但容易转置/leading dimension 出错 | 直接调用 SGEMM 一定比 ATen dispatch 更快 | row-major tensor 用 column-major cuBLAS 解释时很容易把公式写反 | correctness gate 必须先于 benchmark，保留 `max_abs`/`rel_l2` | kernel 优化首先是数学等价，其次才是速度 |
| Phase 2 预计算 `W + A@B^T` 没有成为最终方案 | 理论上把两个阶段合成一次 GEMM，但没有进入 final best | 减少 GEMM 次数就会更快 | materialize `[d,d]` 临时矩阵会增加内存和分配压力，低秩优势被抵消 | 保留 ATen/cuBLAS fallback 和 in-place addmm 作为稳定提交 | 算法重写要算数据移动，不只算公式 |
| Phase 3 hard-coded public config | public toy case 可运行，但 hidden 风险高且吞吐结果不稳定 | hidden 和 public 配置差异不大 | hidden model 的 layer/head/shape 可能变化，权重 key 也必须精确匹配 | prompt 注入 `model_config` 和 state_dict schema，禁止硬编码 | runtime agent 必须从 schema 出发，而不是从样例出发 |
| Phase 3 request batching bug | `torch.tensor(list_of_tensors)` 或 padding 后 last-token logits 错误 | 把 list 转 tensor 是普通 batch 操作 | PyTorch 不接受不同 tensor 对象的这种构造；padding 还会改变 last-token 位置 | 按等长分组，用 `torch.stack`；输出始终取真实 last token | serving runtime 的状态/shape bug 比单次 forward 更隐蔽 |

## 8. What I Would Do Differently

如果从 Phase 1 重新开始，我会先统一一个跨阶段的 artifact schema：每次运行都保存 config、candidate name、source hash、correctness log、benchmark table、提交 output id 和失败原因。Phase 2 的完整 candidate-level `output.json` 没有保留在最终分支里，这是一个明显遗憾；虽然报告和提交信息保留了 best output 和部分 speedup，但复盘其他候选时证据不够完整。

我也会更早写“反例测试”。Phase 1 应该为 profiling workload 加入 sanity check，例如同一 metric 用两个不同 workload 采集并比较数量级；Phase 2 应该保留几个专门抓 transpose/layout 错误的小维度 case；Phase 3 应该从一开始测试不同 prompt length、重复 request id、remove 后复用 id、GQA head ratio、CPU/auto device fallback，以及 prefill 返回 `[batch,vocab]` 而不是 `[batch,seq,vocab]`。

在 agent 架构上，我会把 LLM 只放在“提出候选”和“解释失败”的位置，把验证、评分、状态记录做成更强的 deterministic harness。这样可以减少 prompt 漂移带来的不确定性，也能让每一次失败都成为下一次生成的结构化输入，而不是只靠自然语言总结。

## 9. Conclusion

这个项目最大的技术收获是：MLSys 优化必须同时尊重硬件事实、数学等价和系统 contract。Phase 1 里同一个 metric 可以因为 workload 不同而完全改变含义；Phase 2 里一个看似低秩的公式最后仍受大 GEMM 和内存布局支配；Phase 3 里真正的吞吐来自 cache、batching 和减少 launch overhead，而不只是把 transformer forward 写对。

最大的工程收获是：自动化 agent 的价值不在于一次生成完美答案，而在于把“尝试、验证、失败、修复”变成可重复循环。正确性 gate、结构化日志、输出 shape contract 和 hidden-case 泛化，比单个聪明 prompt 更重要。

我仍然想继续探索的问题是：在不给完整 hidden workload 的情况下，怎样设计更稳健的 public benchmark 和 synthetic traces，让 agent 能更早发现真实 serving 场景中的瓶颈，例如 mixed prefill/decode、KV cache fragmentation 和 batch scheduling 的 trade-off。

## Appendix

### Representative Output IDs

| Phase | Output id |
| --- | --- |
| Phase 1 | `64ee79425df7421281a7f110ac0cb310` |
| Phase 2 | `d558fbf45dbe4dd0749e7dc0430e0bd7` |
| Phase 3 | `04355a648a1278bf94c2dfab3eddbdbb3` |
