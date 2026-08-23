# 公共 Kaggle Soccer 流程

[English](kaggle-soccer.md) | [Kaggle GPU 指南](kaggle.zh-CN.md) | [数据格式](../reference/data-format.zh-CN.md)

公共 [`quantigoai/soccer-dataset`](https://www.kaggle.com/datasets/quantigoai/soccer-dataset) 是一个 CC-BY-SA-4.0 的小型教学数据集，包含三个短视频和 COCO 风格的多边形实例标注。它适合学习数据转换和训练，但不是官方 panoptic benchmark：帧来自视频，数据集没有 provider split，也没有本项目的 crowd 政策。

## 下载

```bash
uv tool install kaggle
kaggle auth login
kaggle datasets download -d quantigoai/soccer-dataset \
  -p data/external/soccer --unzip
```

不要提交下载的压缩包或抽取后的帧。实验记录应保存 Kaggle 数据集引用、下载日期、许可证和源码 revision。

## 转换

转换器只抽取有标注的帧，把多边形栅格化，并生成项目的三目录契约。默认宽度控制本地存储，`--max-frames` 和 `--frame-stride` 控制教学流程规模。

```bash
uv run --with opencv-python-headless python scripts/convert_kaggle_soccer.py \
  data/external/soccer \
  --output data/kaggle-soccer \
  --max-frames 240 \
  --frame-stride 3 \
  --resize-width 512
```

类别映射如下：

| 源类别 | 项目 ID | 类型 |
|---|---:|---|
| Player | 0 | thing |
| Ball | 1 | thing |
| Goal Line | 2 | stuff |
| Field | 3 | stuff |
| Background | 4 | stuff |
| Referee | 5 | thing |
| Football Pitch Line | 6 | stuff |

使用 group-aware split 时，`max-frames` 必须覆盖每个非空 split 至少一个完整 group；文档中的 240 帧示例会覆盖三个视频。只覆盖一个视频的小上限会被拒绝，而不是把该 group 泄漏到多个 split。

转换后准备 manifest：

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

## 训练

```bash
uv run panoptic-segment train --config configs/kaggle_soccer.yaml \
  --set data.data_dir=data/kaggle-soccer \
  --set data.manifest_dir=data/kaggle-soccer \
  --device cuda
uv run panoptic-segment evaluate artifacts/kaggle-soccer-panoptic-unet/best.pt \
  --split valid --device cuda --output artifacts/kaggle-soccer-panoptic-unet/evaluation.json
```

可以用 `--device cpu` 执行 dry-run。该流程的教学顺序是 `download -> convert -> validate -> preview -> train -> evaluate -> inspect failures`。

## Kaggle 执行

把公共数据集挂载到 Kaggle notebook 或私有 kernel。在 kernel 内转换，再对转换后的目录调用通用 runner：

```bash
uv run --with opencv-python-headless python scripts/convert_kaggle_soccer.py \
  /kaggle/input/soccer-dataset \
  --output /kaggle/working/soccer-contract \
  --max-frames 240 --frame-stride 3 --resize-width 512
python scripts/kaggle_train.py \
  --input /kaggle/working/soccer-contract \
  --schema configs/kaggle_soccer_schema.yaml \
  --config configs/kaggle_soccer.yaml
```

runner 会写出 CUDA preflight 日志、resolved artifacts、checkpoint hash、总体 test 指标和 per-class 指标。需要避免重复抽帧时，可以把转换后的输出作为私有 Kaggle Dataset 挂载到下一次运行。

## 协议限制

随机按帧切分可能让同一视频的相邻帧同时进入 train 和 validation。它适合学习流程，但不适合泛化结论。可信实验应在转换前按视频切分，或增加 group-aware manifest splitter。请保留数据集的 CC-BY-SA-4.0 attribution，不要把内部 non-crowd PQ 称为官方 benchmark 分数。
