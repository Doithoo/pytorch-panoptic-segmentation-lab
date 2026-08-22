# Checkpoint Schema v1

[English](checkpoint-schema.md) | [训练教程](../tutorial/04-training.zh-CN.md)

只接受 schema version 1。保存先写唯一临时文件，再用 `os.replace` 发布；加载调用 `torch.load(..., weights_only=True)`，拒绝字段缺失或版本不兼容的 mapping。

## 顶层字段

| 字段 | 含义 |
|---|---|
| `schema_version` | 整数 `1` |
| `config` | 完整 resolved config |
| `schema` | 有序类别、thing、颜色、ignore |
| `model_state` | tensor state dict |
| `optimizer_state` | optimizer 参数组与 tensor |
| `scheduler_state` | scheduler mapping 或 null |
| `scaler_state` | CUDA GradScaler mapping，可为空 |
| `epoch` | 最近完成 epoch |
| `best_metric` | 截止该轮的最佳 validation 值 |
| `metrics` | 完整 CSV row mapping 列表 |
| `run_metadata` | Python/torch/平台/设备/CUDA/seed/Git revision |
| `dataset_identity` | prepared schema/manifest identity |
| `rng_state` | Python、NumPy、torch、可用 CUDA 状态 |

预测只需 config/schema/model；恢复还需要 optimizer、scheduler、scaler、RNG、metrics 和 identity。

恢复要求 schema、数据身份、model、loss、后处理、optimizer/scheduler/seed/AMP/梯度设置，以及尺寸、batch、增强、sigma、样本上限等数据语义一致。可增加 `train.epochs`，可改变数据/manifest 路径、worker 和 device；但必须使用配置运行目录内的 `last.pt`，以保持历史和旧 best 一致。

目标 metrics 会追加。已有运行目录中启动普通训练会失败，防止静默覆盖历史。

请使用项目加载器，不可信 checkpoint 不要回退 `weights_only=False`。未来 external factory 必须单独定义代码信任边界和 checkpoint schema 决策。
