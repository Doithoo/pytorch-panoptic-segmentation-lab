# 公共 Kaggle Soccer 数据

[English](kaggle-soccer.md) | [Kaggle GPU](kaggle.zh-CN.md) | [数据格式](../reference/data-format.zh-CN.md)

公共 [`quantigoai/soccer-dataset`](https://www.kaggle.com/datasets/quantigoai/soccer-dataset) 包含三个短视频和 COCO 风格的多边形标注。它的规模适合第一次真实数据运行，也能让你看到模型示例通常省略的工作：读取标注格式、抽取视频帧、把多边形栅格化、确定类别含义，以及决定如何划分数据。

该数据集使用 CC-BY-SA-4.0。请保留 attribution，不要提交下载包和抽取后的帧。

## 下载

```bash
uv tool install kaggle
kaggle auth login
kaggle datasets download -d quantigoai/soccer-dataset \
  -p data/external/soccer --unzip
```

## 转换标注

转换器会抽取有标注的帧，并生成：

```text
data/kaggle-soccer/
  images/
  semantic/
  instance/
  groups.csv
  schema.yaml
  source.json
```

学习时可以先限制输出宽度和帧数。下面的命令每隔 5 帧抽取一次，并覆盖三个源视频：

```bash
uv run --with opencv-python-headless python scripts/convert_kaggle_soccer.py \
  data/external/soccer \
  --output data/kaggle-soccer \
  --max-frames 240 \
  --frame-stride 5 \
  --resize-width 512
```

源类别会按下面的方式映射：

| 源类别 | ID | 模型中的含义 |
|---|---:|---|
| Player | 0 | thing |
| Ball | 1 | thing |
| Goal Line | 2 | stuff |
| Field | 3 | stuff |
| Background | 4 | stuff |
| Referee | 5 | thing |
| Football Pitch Line | 6 | stuff |

一个 group 对应一个源视频。如果帧数上限只覆盖一个视频，按组切分就无法生成三个非空 split，程序会停止。这是为了防止相邻帧同时进入训练集和验证集。

## 准备并检查

```bash
uv run panoptic-segment prepare-data \
  --data-dir data/kaggle-soccer \
  --manifest-dir data/kaggle-soccer \
  --group-file data/kaggle-soccer/groups.csv \
  --schema configs/kaggle_soccer_schema.yaml
uv run panoptic-segment inspect-data --manifest-dir data/kaggle-soccer
uv run python scripts/preview_panoptic.py data/kaggle-soccer/train.csv \
  --output artifacts/soccer-preview.png --limit 4
```

训练前打开预览图，确认球场和背景颜色正确，球员使用正整数 instance ID，并查看 `dataset.yaml` 中是否每个 split 只包含一个视频 group。

## 在 GPU 上训练

```bash
uv run panoptic-segment train --config configs/kaggle_soccer.yaml \
  --set data.data_dir=data/kaggle-soccer \
  --set data.manifest_dir=data/kaggle-soccer \
  --device cuda
uv run panoptic-segment evaluate artifacts/kaggle-soccer-panoptic-unet/best.pt \
  --split valid --device cuda \
  --output artifacts/kaggle-soccer-panoptic-unet/evaluation.json
```

只检查本地流程时，把 `--device cuda` 换成 `--device cpu`，并在训练命令上加 `--dry-run`。

## 在 Kaggle 上运行

仓库在 `docs/recorded-run/kaggle-soccer/` 中提供了可以直接提交的 kernel。它会挂载公开数据集，执行转换，把 `groups.csv` 传给 manifest 生成器，在 T4 上训练，并保存逐图评估结果。

```bash
kaggle kernels push -p docs/recorded-run/kaggle-soccer
kaggle kernels status yashowhoo/pytorch-panoptic-segmentation-lab-kaggle-soccer
```

已记录的 version 2 使用 train `Batch 3`、validation `Batch 1`、test `Batch 2`，训练十轮后 validation PQ 为 `0.290397`，test PQ 为 `0.223444`。thing PQ 很低，适合从 center loss、图像尺寸、后处理阈值或模型宽度开始尝试改动。

## 这个结果不能说明什么

该数据集没有官方 panoptic leaderboard 协议，三个视频 group 也不足以支撑强泛化结论。这次运行说明的是转换和训练路径可以工作，不代表模型在所有 Soccer 视频上的表现。
