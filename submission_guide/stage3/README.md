# MLSys Phase 3 Public Skeleton

本目录是第三阶段的最小公开样例。它模拟了学生 agent 运行完毕之后，评测系统看到的仓库状态。

目录结构：

```text
mlsys/
  README.md
  PHASE3_STUDENT_GUIDE.md
  run.sh
  agent.py
  target/
    model_config.json
    weights/model.pt
  workspace/
    engine.py
    results.log
  evaluator/
    reference_model.py
    test_correctness.py
    benchmark_throughput.py
  scripts/
    generate_toy_weights.py
    run_public_tests.sh
```

公开样例的核心命令：

```bash
python3 scripts/generate_toy_weights.py \
  --config target/model_config.json \
  --output target/weights/model.pt

python3 evaluator/test_correctness.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto

python3 evaluator/benchmark_throughput.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto
```

真实评测时，测试代码结构与这里一致，但会替换隐藏的模型规格、权重和请求 trace。

如果你的默认 `python3` 没有 PyTorch，可以这样指定解释器：

```bash
PYTHON=/path/to/python-with-torch bash scripts/run_public_tests.sh
```
