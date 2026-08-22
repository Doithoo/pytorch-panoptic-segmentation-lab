# 全景分割学习路线

[English](learning-path.md) | [文档中心](../README.zh-CN.md)

本路线约需 8–12 小时，假设你已理解基础 Tensor、卷积和梯度下降。完成每条命令并能解释输出后，再开始 Kaggle 任务。

## 1. 验证环境

```bash
uv sync --extra dev
uv run panoptic-segment --version
uv run panoptic-segment show-config --config configs/learning_minimal.yaml
make check
```

从 resolved config 中找出输入尺寸、样本上限、三项 loss 权重、后处理上限、设备和最佳指标。

## 2. 检查 target 契约

```bash
uv run python scripts/create_synthetic_data.py --count 24 --size 128
uv run panoptic-segment prepare-data --schema configs/synthetic_schema.yaml
uv run panoptic-segment inspect-data
uv run python examples/01_panoptic_target.py
```

解释 semantic `[H,W]`、center `[H,W]`、offset `[2,H,W]`，以及 instance 为何不直接输入模型。确认 thing 像素使用正 instance ID，stuff 和 void 使用 0。

## 3. 跟踪一次参数更新

```bash
uv run python examples/02_model_contract.py
uv run panoptic-segment train --config configs/learning_minimal.yaml --dry-run
```

Dry run 会执行 forward、有限 loss 检查、backward、梯度裁剪和一次 optimizer step，但不会发布正常 checkpoint 或指标。

## 4. 完成并检查一次运行

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml
```

依次阅读 `config.yaml`、`run.yaml`、`metrics.csv`、`best.pt`、`last.pt`。解释 best 与 last 为何可能不同，以及为何 test 不能参与模型选择。

## 5. 从指标回到像素

```bash
uv run panoptic-segment evaluate artifacts/learning-minimal/best.pt --split test
uv run panoptic-segment predict artifacts/learning-minimal/best.pt \
  data/raw/images/sample_0000.png
```

对比总体 PQ、`pq_thing`、`pq_stuff`，再查看配色 mask 和 overlay。低 PQ 可能来自语义错误、中心漏检/重复、offset 偏差或面积过滤。

## 6. 恢复一个受控实验

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml \
  --set train.epochs=4 --resume artifacts/learning-minimal/last.pt
```

恢复会还原 model、optimizer、scheduler、scaler、RNG、指标历史和 best 值。改变模型、loss、后处理、schema 或数据身份会被拒绝。

## 7. 提交 Kaggle 参考任务

按照 [Kaggle 指南](../guides/kaggle.zh-CN.md)操作。首次任务使用确定性合成数据证明非交互 GPU 流程，属于系统证据，不是真实世界 benchmark。

当你能解释 Gaussian center、thing-only offset、void、按类别累计 PQ、验证集选择、安全 checkpoint，以及流程记录与 benchmark 声明的区别时，就可以开始扩展项目。
