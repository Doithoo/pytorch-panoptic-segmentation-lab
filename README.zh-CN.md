# PyTorch Panoptic Segmentation Lab

[English](README.md) | [文档中心](docs/README.zh-CN.md) | [Kaggle 指南](docs/guides/kaggle.zh-CN.md)

一个强调可读性与实验可复现性的 PyTorch 全景分割项目。基线模型同时预测语义类别、thing 中心热图和像素到中心的 offset，再把结果组合为 thing 实例与 stuff 区域。

> 项目状态：本地流程和安装包流程已有测试覆盖，确定性合成数据 Kaggle 参考任务也已成功完成。指标和证据见[参考运行](docs/recorded-run/README.zh-CN.md)。该结果是流程证据而不是真实数据 benchmark；内置 PQ 评估器适用于本项目“不含 crowd”的 mask 契约，不能替代具体数据集的 crowd 规则或官方评测服务器。

## 已实现能力

- semantic、center heatmap、offset 三头 Panoptic U-Net。
- 同步几何变换和 Gaussian 中心监督。
- thing-only offset mask 与稀疏关键点 focal center loss。
- 有中心数量上限、类别约束和面积过滤的后处理。
- 先按类别累计再宏平均的 PQ/SQ/RQ，以及 thing/stuff 分项。
- 确定性 manifest、数据身份和全景标签预检。
- `weights_only=True`、原子写入、带版本且可恢复的 checkpoint。
- 完整配置、训练历史、运行环境、Git revision 和数据身份记录。
- 原始 semantic/instance mask、语义配色图和全景 overlay。
- CPU 自动测试和带心跳、自动评估的 Kaggle T4 runner。

## 快速开始

```bash
uv sync --extra dev
uv run python scripts/create_synthetic_data.py
uv run panoptic-segment prepare-data --schema configs/synthetic_schema.yaml
uv run panoptic-segment inspect-data
uv run panoptic-segment train --config configs/learning_minimal.yaml --dry-run
uv run panoptic-segment train --config configs/learning_minimal.yaml
```

用验证集选出的 checkpoint 评估和预测：

```bash
uv run panoptic-segment evaluate artifacts/learning-minimal/best.pt --split test
uv run panoptic-segment predict artifacts/learning-minimal/best.pt \
  data/raw/images/sample_0000.png --output artifacts/prediction
```

一次预测会输出：

```text
sample_0000.semantic.png
sample_0000.instance.png
sample_0000.semantic-color.png
sample_0000.overlay.png
```

## 数据契约

每个样本由三份同 stem 文件构成：

```text
data/raw/
  images/sample_0001.png
  semantic/sample_0001.png   # 连续 class ID 或 255
  instance/sample_0001.png   # stuff/void 为 0，thing 为正整数
```

每个正 instance ID 在一张图中只能对应一个 thing 类。thing 像素必须有正 instance ID，stuff 和 ignore 像素必须为 0。`prepare-data` 会拒绝 stem 不匹配并生成使用相对路径的 manifest；`inspect-data` 会检查解码、尺寸、标签、实例关系、split 数量和跨 split 重复 ID。

适配真实数据前请先阅读[数据格式参考](docs/reference/data-format.zh-CN.md)。

## 训练产物

正常训练写入 `artifacts/<run_name>/`：

| 文件 | 作用 |
|---|---|
| `config.yaml` | 本次运行实际使用的完整配置 |
| `run.yaml` | Python/PyTorch/平台/设备、Git revision、数据身份和时间 |
| `metrics.csv` | loss 分量、学习率、PQ/SQ/RQ、thing/stuff PQ |
| `last.pt` | 最新模型以及 optimizer、scheduler、scaler、RNG、历史 |
| `best.pt` | 按 `train.best_metric` 选出的 checkpoint |

只允许恢复契约兼容的实验：

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml \
  --set train.epochs=4 --resume artifacts/learning-minimal/last.pt
```

checkpoint 使用安全、带版本的加载器。对不可信文件不要绕开项目去调用 `weights_only=False`。

## 配置系统

YAML 采用严格模式，未知字段会报错而不是静默忽略。命令行可覆盖单个值：

```bash
uv run panoptic-segment show-config --config configs/learning_minimal.yaml \
  --set data.batch_size=4 --set run_name=experiment-01
```

输入高宽必须被 16 整除；schema 类别数和 ignore index 必须与 model/loss 一致。后处理参数属于保存配置的一部分，确保评估和预测遵守同一契约。

完整字段见[配置参考](docs/reference/config-reference.zh-CN.md)。

## Kaggle GPU

仓库提供一个不依赖外部 Dataset 的合成数据参考 kernel，用来证明源码获取、CUDA kernel、训练、checkpoint 重载、test 评估和产物导出能在一次非交互 Kaggle 任务中完成。

```bash
uv tool install kaggle
kaggle auth login
# 修改 docs/recorded-run/kaggle/kernel-metadata.json 中的账号
kaggle kernels push -p docs/recorded-run/kaggle
```

请选择 T4 或更新的 NVIDIA GPU。runner 会记录实际 Git commit 和 checkpoint SHA-256。发布 Cityscapes 或 COCO 真实结果前，还必须补充数据转换器、官方 split、数据集特定的 crowd/void 行为，并遵守数据许可。

完整流程见 [Kaggle 指南](docs/guides/kaggle.zh-CN.md)。

## 学习路径

1. [Tensor 与 panoptic ID](docs/tutorial/00-basics.zh-CN.md)
2. [环境和 CLI](docs/tutorial/01-environment.zh-CN.md)
3. [数据、中心热图与 offset](docs/tutorial/02-data-and-targets.zh-CN.md)
4. [Panoptic U-Net](docs/tutorial/03-panoptic-unet.zh-CN.md)
5. [训练、产物和断点恢复](docs/tutorial/04-training.zh-CN.md)
6. [评估、预测与边界](docs/tutorial/05-evaluation-and-inference.zh-CN.md)

可从[完整学习路线](docs/tutorial/learning-path.zh-CN.md)开始，阅读源码时配合[代码导览](docs/concepts/code-tour.zh-CN.md)。

## 范围与限制

当前基线刻意保持小型并从头训练。它用于展示完整工程契约，不宣称与原始 Panoptic-DeepLab 架构等价，也不宣称达到先进精度。项目当前没有预训练 backbone、官方 Cityscapes/COCO 转换器、crowd 字段、分布式训练或已完成的真实数据 Kaggle 记录。这些是明确的扩展方向，不是隐藏能力。

运行全部质量门禁：

```bash
make check
```

贡献应保持学习路径可读，为指标提供可手算测试，并同步记录新增 target/checkpoint 字段。详见 [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md)。
