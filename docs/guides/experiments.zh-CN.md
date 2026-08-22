# 受控实验

[English](experiments.md) | [配置流](../concepts/configuration-flow.zh-CN.md)

每次只改变一个解释变量，并使用唯一 run name：

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml \
  --set run_name=center-threshold-015 \
  --set postprocess.center_threshold=0.15
```

除目标变量外，保持 data identity、split、seed、模型、epoch 和选择指标不变。比较实际 `config.yaml`，不要依赖记忆中的命令。报告最佳 validation epoch、选择指标、最终 test、环境、样本上限和至少一个失败可视化。

在 validation 调后处理也属于模型选择。test 前冻结阈值；反复在 test 上试阈值会泄漏信息，使结果虚高。

做一般性优化结论前至少运行三个 seed。2 epoch 合成对比只证明流程行为，应记录均值、波动、耗时和失败，而不只保留赢家。
