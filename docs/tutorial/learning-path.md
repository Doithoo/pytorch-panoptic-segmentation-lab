# Panoptic Segmentation Learning Path

[简体中文](learning-path.zh-CN.md) | [Documentation](../README.md)

This path takes about 8-12 hours for someone who knows basic tensors, convolutions, and gradient descent. It starts with generated images, then moves to a public annotated video dataset. Stop after any step if you want to inspect the code in more detail.

## 1. Get a working environment

```bash
uv sync --locked --extra dev
uv run panoptic-segment --version
uv run panoptic-segment show-config --config configs/learning_minimal.yaml
make check
```

Open the printed configuration and find the input size, sample limits, loss weights, post-processing thresholds, device, and metric used to choose `best.pt`.

## 2. Look at the labels before the model

```bash
uv run python scripts/create_synthetic_data.py --count 24 --size 128
uv run panoptic-segment prepare-data --schema configs/synthetic_schema.yaml
uv run panoptic-segment inspect-data
uv run python scripts/preview_panoptic.py data/manifests/train.csv \
  --output artifacts/learning-preview.png --limit 4
uv run python examples/01_panoptic_target.py
```

Answer these questions:

- What does `semantic[y, x]` store?
- Why is a thing pixel required to have a positive instance ID?
- Why does the instance mask not go into the model as an input?
- What do the center heatmap and the two offset channels point to?

Do not continue until the preview colors, object boundaries, and instance IDs make sense.

## 3. Run one update

```bash
uv run python examples/02_model_contract.py
uv run panoptic-segment train --config configs/learning_minimal.yaml --dry-run
```

Read the output shapes, then trace the call from the CLI to the model, loss, backward pass, and optimizer step. The dry run is a connection check; it is not a quality measurement.

## 4. Train and inspect the files

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml
```

Open these files in order:

1. `config.yaml`: what settings were used;
2. `run.yaml`: which software, device, revision, and data fingerprint were recorded;
3. `metrics.csv`: how loss and validation metrics changed;
4. `best.pt`: the validation-selected checkpoint;
5. `last.pt`: the latest checkpoint used for resume.

Explain why `best.pt` and `last.pt` can differ, and why the test split must not choose the checkpoint.

## 5. Compare numbers with pixels

```bash
uv run panoptic-segment evaluate artifacts/learning-minimal/best.pt --split test \
  --output artifacts/learning-minimal/evaluation.json
uv run panoptic-segment predict artifacts/learning-minimal/best.pt \
  data/raw/images/sample_0000.png --output artifacts/prediction
```

Compare `pq`, `pq_thing`, and `pq_stuff`. Then inspect the semantic-color and overlay images. When PQ is low, look separately for wrong classes, missing or duplicate centers, bad offsets, merged objects, split objects, and area filtering.

## 6. Change one thing and resume

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml \
  --set train.epochs=4 --resume artifacts/learning-minimal/last.pt
```

The checkpoint restores model and optimizer state, scheduler state, random-number state, metric history, and the previous best value. It rejects changes that would make the resumed run use different labels, targets, post-processing, or data membership.

## 7. Use a real public dataset

Follow [Kaggle Soccer](../guides/kaggle-soccer.md). The converter starts with videos and COCO polygons, extracts frames, writes semantic and instance masks, creates a split by source video, and then reuses the same training commands.

The recorded run reaches test PQ `0.223444`; thing PQ is `0.000000`. That result is useful because it shows what a small from-scratch model fails to learn. Read the per-class file and worst-case report before changing the model.

## 8. Extend the project

Once you can trace one sample from source annotation to prediction image, try one change:

- add a class to the schema;
- change the data converter;
- register another model;
- adjust one post-processing value;
- compare two runs with the same split and seed.

Use the guides and reference pages when you need exact field names or compatibility rules.
