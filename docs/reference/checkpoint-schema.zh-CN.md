# Checkpoint Schema v1

[English](checkpoint-schema.md) | [训练](../tutorial/04-training.zh-CN.md)

项目只接受 schema version 1。保存时先写临时文件，再用 `os.replace` 替换目标文件。加载使用 `torch.load(..., weights_only=True)`，缺少字段或字段版本不兼容时会拒绝加载。

## 字段

| 字段 | 含义 |
|---|---|
| `schema_version` | 整数 `1` |
| `config` | 本次实验的最终设置 |
| `schema` | 有序类别、thing 标记、颜色和 ignore ID |
| `model_state` | Tensor state dict |
| `optimizer_state` | optimizer 状态 |
| `scheduler_state` | scheduler 状态或 null |
| `scaler_state` | CUDA scaler 状态，可能为空 |
| `epoch` | 最近完成的 epoch |
| `best_metric` | 当前为止最好的验证指标 |
| `metrics` | 写入 `metrics.csv` 的各行记录 |
| `run_metadata` | Python、torch、torchvision、平台、设备、seed 和 Git revision |
| `dataset_identity` | prepared schema 和 manifest 的数据指纹 |
| `rng_state` | 可用时保存 Python、NumPy、torch 和 CUDA 随机数状态 |

预测需要 `config`、`schema` 和 `model_state`。恢复训练还需要 optimizer、scheduler、scaler、随机数状态、训练历史和数据指纹。

## 恢复规则

恢复时必须保持 schema、数据指纹、模型、loss、后处理、optimizer、scheduler、seed、增强、target sigma、图像尺寸、batch size 和样本上限一致。可以增加 `train.epochs`。数据路径、worker 数量和设备属于运行参数，可以改变。

恢复必须使用配置运行目录中的 `last.pt`。普通训练不会覆盖已有的 metrics 文件。

## 不可信文件

使用项目加载器：

```python
from panoptic_segmenter.training.checkpoint import load_checkpoint

checkpoint = load_checkpoint("artifacts/run/best.pt")
```

对于不是自己创建或检查过的 checkpoint，不要切换到 `weights_only=False`。模型重建会执行保存配置中指定的、当前环境里已安装的模型 factory。
