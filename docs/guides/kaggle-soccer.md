# Public Kaggle Soccer Workflow

[简体中文](kaggle-soccer.zh-CN.md) | [Kaggle GPU guide](kaggle.md) | [Data format](../reference/data-format.md)

The public [`quantigoai/soccer-dataset`](https://www.kaggle.com/datasets/quantigoai/soccer-dataset) is a small teaching dataset under CC-BY-SA-4.0. It contains three short videos and COCO-style polygon instance annotations. It is useful for learning conversion and training, but it is not an official panoptic benchmark: frames come from videos, splits are not provider-defined, and the dataset has no project-specific crowd policy.

## Download

```bash
uv tool install kaggle
kaggle auth login
kaggle datasets download -d quantigoai/soccer-dataset \
  -p data/external/soccer --unzip
```

Do not commit the downloaded archive or extracted frames. Record the Kaggle dataset reference, download date, license, and source revision in your experiment notes.

## Convert

The converter extracts only annotated frames, rasterizes polygons, and writes the project's three-folder contract. The default width keeps local storage manageable; `--max-frames` and `--frame-stride` make the teaching run bounded.

```bash
uv run --with opencv-python-headless python scripts/convert_kaggle_soccer.py \
  data/external/soccer \
  --output data/kaggle-soccer \
  --max-frames 240 \
  --frame-stride 3 \
  --resize-width 512
```

The mapping is:

| Source class | Project ID | Kind |
|---|---:|---|
| Player | 0 | thing |
| Ball | 1 | thing |
| Goal Line | 2 | stuff |
| Field | 3 | stuff |
| Background | 4 | stuff |
| Referee | 5 | thing |
| Football Pitch Line | 6 | stuff |

With group-aware splitting, `max-frames` must include at least one complete group for every non-empty split; the documented 240-frame example samples all three videos. A tiny limit that covers only one video is rejected rather than leaking that group across splits.

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

## Train

```bash
uv run panoptic-segment train --config configs/kaggle_soccer.yaml \
  --set data.data_dir=data/kaggle-soccer \
  --set data.manifest_dir=data/kaggle-soccer \
  --device cuda
uv run panoptic-segment evaluate artifacts/kaggle-soccer-panoptic-unet/best.pt \
  --split valid --device cuda --output artifacts/kaggle-soccer-panoptic-unet/evaluation.json
```

Use `--device cpu` for a dry-run. This workflow is intended to teach the sequence `download -> convert -> validate -> preview -> train -> evaluate -> inspect failures`.

## Kaggle execution

Attach the public dataset to a Kaggle notebook or private kernel. Convert it inside the kernel, then call the generic runner on the converted directory:

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

The runner writes CUDA preflight logs, resolved artifacts, checkpoint hashes, aggregate test metrics, and per-class metrics. Attach the converted output as a private Kaggle Dataset when you need a second run without repeating extraction.

## Protocol limits

Random frame splitting can put adjacent frames from the same video in train and validation. That is acceptable for learning the mechanics but invalid for a generalization claim. For a credible experiment, split by video before conversion or add a group-aware manifest splitter. Preserve the dataset's CC-BY-SA-4.0 attribution and do not call the internal non-crowd PQ an official benchmark score.
