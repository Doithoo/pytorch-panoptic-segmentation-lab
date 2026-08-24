# Public Kaggle Soccer Data

[简体中文](kaggle-soccer.zh-CN.md) | [Kaggle GPU](kaggle.md) | [Data format](../reference/data-format.md)

The public [`quantigoai/soccer-dataset`](https://www.kaggle.com/datasets/quantigoai/soccer-dataset) contains three short videos and COCO-style polygon annotations. It is small enough for a first real-data run and exposes the work that a model-only example usually skips: reading an annotation format, extracting frames, rasterizing polygons, deciding class meanings, and choosing a split.

The dataset is licensed CC-BY-SA-4.0. Keep the attribution, and do not commit the download or extracted frames.

## Download

```bash
uv tool install kaggle
kaggle auth login
kaggle datasets download -d quantigoai/soccer-dataset \
  -p data/external/soccer --unzip
```

## Convert the annotations

The converter extracts annotated frames and writes:

```text
data/kaggle-soccer/
  images/
  semantic/
  instance/
  groups.csv
  schema.yaml
  source.json
```

Use a small output width and a frame limit while learning. The example below samples every fifth annotated frame and covers all three source videos:

```bash
uv run --with opencv-python-headless python scripts/convert_kaggle_soccer.py \
  data/external/soccer \
  --output data/kaggle-soccer \
  --max-frames 240 \
  --frame-stride 5 \
  --resize-width 512
```

The converter maps the source classes as follows:

| Source | ID | Model meaning |
|---|---:|---|
| Player | 0 | thing |
| Ball | 1 | thing |
| Goal Line | 2 | stuff |
| Field | 3 | stuff |
| Background | 4 | stuff |
| Referee | 5 | thing |
| Football Pitch Line | 6 | stuff |

A group is the source video. If the frame limit covers only one video, the grouped splitter cannot create three non-empty splits and will stop. That is deliberate: it prevents adjacent frames from appearing in both training and validation.

## Prepare and inspect

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

Check the preview before training. Confirm that field and background regions have the expected colors, players have positive instance IDs, and the group list in `dataset.yaml` contains one video per split.

## Train on a GPU

```bash
uv run panoptic-segment train --config configs/kaggle_soccer.yaml \
  --set data.data_dir=data/kaggle-soccer \
  --set data.manifest_dir=data/kaggle-soccer \
  --device cuda
uv run panoptic-segment evaluate artifacts/kaggle-soccer-panoptic-unet/best.pt \
  --split valid --device cuda \
  --output artifacts/kaggle-soccer-panoptic-unet/evaluation.json
```

For a CPU connection check, replace `--device cuda` with `--device cpu` and add `--dry-run` to the training command.

## Run it on Kaggle

The repository includes a ready-to-submit kernel under `docs/recorded-run/kaggle-soccer/`. It attaches the public dataset, runs the conversion, passes `groups.csv` to the manifest builder, trains on a T4, and saves per-image evaluation output.

```bash
kaggle kernels push -p docs/recorded-run/kaggle-soccer
kaggle kernels status yashowhoo/pytorch-panoptic-segmentation-lab-kaggle-soccer
```

The recorded version 2 run used train `Batch 3`, validation `Batch 1`, and test `Batch 2`. It reached validation PQ `0.290397` and test PQ `0.223444` after ten epochs. Its low thing PQ is a useful starting point for changing the center loss, image size, post-processing thresholds, or model width.

## What this result does not show

This dataset has no official panoptic leaderboard protocol. Three video groups are also too few for a strong generalization claim. The run demonstrates the conversion and training path; it does not establish performance on Soccer video in general.
