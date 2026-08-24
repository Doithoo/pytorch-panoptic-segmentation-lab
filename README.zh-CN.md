# PyTorch 全景分割

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Doithoo/pytorch-panoptic-segmentation-lab/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

[English](README.md) | [文档目录](docs/README.zh-CN.md) | [Kaggle Soccer](docs/guides/kaggle-soccer.zh-CN.md) | [Cityscapes](docs/guides/cityscapes.zh-CN.md)

通过一个可以直接运行和修改的 PyTorch 项目学习全景分割。模型预测语义类别、对象中心和像素到中心的偏移量，解码器再把这些输出组合为可数对象（thing）和不可数区域（stuff）。

![合成数据原图、语义标签和全景叠加图](docs/assets/synthetic-panoptic-preview.png)

这里不只提供模型代码，还包括标签转换、数据划分、训练前检查、同步数据增强、监督目标构建、训练、断点恢复、PQ 评估、单图预测和运行记录。

## 选择一条路线

| 路线 | 数据 | 适合解决的问题 |
|---|---|---|
| [合成数据快速开始](#合成数据快速开始) | 本地生成 | 在 CPU 上看懂张量，并确认整条流程可以运行 |
| [Kaggle Soccer](docs/guides/kaggle-soccer.zh-CN.md) | 公开视频和多边形标注 | 学习如何把原始标注转换为 mask、按视频划分数据并在 GPU 上训练 |
| [Cityscapes](docs/guides/cityscapes.zh-CN.md) | 需要接受许可条款的官方数据 | 学习 train ID 映射、官方数据划分、crowd 处理和官方评估 |
| [自己的数据](docs/guides/using-your-data.zh-CN.md) | 自备图像和标签 | 调整数据格式和类别定义 |

合成数据和 Soccer 的实测结果保存在 [`docs/recorded-run/`](docs/recorded-run/) 中。这些记录用于复现命令和分析输出，不代表 Cityscapes 或 COCO 排行榜成绩。

## 合成数据快速开始

准备 Python 3.10-3.12 和 [uv](https://docs.astral.sh/uv/)，然后执行：

```bash
uv sync --locked --extra dev
uv run python scripts/create_synthetic_data.py
uv run panoptic-segment prepare-data --schema configs/synthetic_schema.yaml
uv run panoptic-segment inspect-data
uv run python scripts/preview_panoptic.py data/manifests/train.csv \
  --output artifacts/dataset-preview.png --limit 4
uv run panoptic-segment train --config configs/learning_minimal.yaml --dry-run
uv run panoptic-segment train --config configs/learning_minimal.yaml
```

`--dry-run` 会实际执行前向计算、损失计算、反向传播、梯度裁剪和一次参数更新，但不会创建正式运行目录。两轮训练的结果写入 `artifacts/learning-minimal/`。

评估验证集选出的 checkpoint，并对一张图像进行预测：

```bash
uv run panoptic-segment evaluate artifacts/learning-minimal/best.pt --split test \
  --output artifacts/learning-minimal/evaluation.json
uv run panoptic-segment predict artifacts/learning-minimal/best.pt \
  data/raw/images/sample_0000.png --output artifacts/prediction
```

预测会同时保存可供程序读取的 mask 和便于观察的图像：

```text
sample_0000.semantic.png
sample_0000.instance.png
sample_0000.semantic-color.png
sample_0000.overlay.png
```

## 数据格式

每个样本包含一张图像和两张同名 mask：

```text
data/raw/
  images/sample_0001.png
  semantic/sample_0001.png   # 连续类别 ID，或 255
  instance/sample_0001.png   # thing 使用正整数；stuff 和 void 使用 0
```

在同一张图中，一个正整数 instance ID 只能属于一个 thing 类。thing 像素必须有正整数 instance ID，stuff 和忽略像素使用 0。`prepare-data` 负责配对文件并生成固定的数据清单；`inspect-data` 会检查图像解码、尺寸、类别 ID、实例 ID、划分数量、文件哈希和分组泄漏。

如果数据来自视频帧、相邻裁剪或同一场景，请向 `prepare-data --group-file` 传入包含 `sample_id,group_id` 的 CSV，确保相关样本不会被拆到不同数据集。细节见[数据格式参考](docs/reference/data-format.zh-CN.md)。

## 更换数据或模型

使用新数据集时：

1. 把标签转换为上面的三目录格式。
2. 在 schema YAML 中定义连续类别 ID、显示颜色和 `isthing`。
3. 数据集有官方划分时沿用官方划分；样本之间有关联时按组划分。
4. 训练前运行 `inspect-data`，并打开预览图人工检查。
5. 设置 `data.manifest_dir`、`model.expected_num_classes` 和 `loss.ignore_index`。

先阅读[使用自己的数据](docs/guides/using-your-data.zh-CN.md)。需要编写转换器时，再阅读[添加数据集](docs/guides/adding-datasets.zh-CN.md)。

替换模型时，新模型必须返回：

```text
semantic [B,C,H,W]
center   [B,1,H,W]
offset   [B,2,H,W]
```

使用 `register_model()` 注册模型构造函数，添加对应配置，并测试 CPU 前向与反向计算。完整步骤见[添加模型](docs/guides/adding-models.zh-CN.md)。

## 训练输出

每次正式训练都会创建 `artifacts/<run_name>/`：

| 文件 | 内容 |
|---|---|
| `config.yaml` | 合并默认值、YAML 和命令行参数后的最终配置 |
| `run.yaml` | Python、PyTorch、设备、Git revision、数据指纹和运行时间 |
| `metrics.csv` | 各项损失、学习率、验证集 PQ/SQ/RQ、thing PQ 和 stuff PQ |
| `last.pt` | 最新模型及 optimizer、scheduler、scaler、随机数状态和训练历史 |
| `best.pt` | 按 `train.best_metric` 选出的 checkpoint |

增加训练轮数并从同一次运行继续：

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml \
  --set train.epochs=4 --resume artifacts/learning-minimal/last.pt
```

加载器使用 `torch.load(..., weights_only=True)`，并检查 checkpoint 版本、模型配置、类别定义和数据指纹。不要对来源不可信的 checkpoint 关闭 `weights_only`。

## 已记录的运行结果

| 运行 | 数据和划分 | 结果 |
|---|---|---|
| [合成数据 T4 运行](docs/recorded-run/README.zh-CN.md) | 256 张生成图像，固定划分为 205/26/25 | test PQ `0.853881` |
| [Kaggle Soccer T4 运行](docs/recorded-run/kaggle-soccer/README.zh-CN.md) | CC-BY-SA-4.0 公开数据，按源视频划分 | test PQ `0.223444`，thing PQ `0.000000`，stuff PQ `0.391027` |

Soccer 的对象分离效果很差，但这正是值得保留的结果。按类别和按图统计可以清楚看到：模型很快学会了球场和背景，却没有可靠地区分球员、足球和裁判。

## 文档入口

| 你想了解什么 | 文档 |
|---|---|
| 应该从哪里开始？ | [教程目录](docs/tutorial/README.zh-CN.md)或[学习路线](docs/tutorial/learning-path.zh-CN.md) |
| 一个样本如何经过各个模块？ | [运行流程](docs/concepts/how-it-works.zh-CN.md)和[代码导览](docs/concepts/code-tour.zh-CN.md) |
| 命令有哪些参数？ | [CLI 参考](docs/reference/cli.zh-CN.md) |
| 配置字段有什么作用？ | [配置参考](docs/reference/config-reference.zh-CN.md) |
| 运行失败时如何排查？ | [故障排查](docs/guides/troubleshooting.zh-CN.md) |

## 仓库结构

```text
configs/                     可直接运行的实验配置
docs/tutorial/               按学习顺序组织的概念与实践
docs/guides/                 常见任务的操作步骤
docs/reference/              数据格式、配置、指标和 checkpoint 结构
docs/recorded-run/           实测运行记录和小型结果文件
examples/                    监督目标和模型输出的短程序
scripts/                     转换、预览、评估和 Kaggle 命令
src/panoptic_segmenter/      可安装的 Python 包
tests/                       离线单元测试和端到端测试
```

## 当前范围

仓库内置的是一个从头训练的小型 U-Net，适合跟踪完整流程和尝试受控修改，但它不是完整的 Panoptic-DeepLab 实现，也不以当前基准精度为目标。项目暂不包含预训练 backbone 和分布式训练。只有遵守 Cityscapes 或 COCO 的数据规则并使用官方评估器后，相关分数才适合与公开结果比较。

运行全部本地检查：

```bash
make check
```

提交改动前请阅读 [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md)。安全问题请按 [SECURITY.md](SECURITY.md) 私下报告，不要公开提交 issue。
