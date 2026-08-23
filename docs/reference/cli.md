# CLI Reference

[简体中文](cli.zh-CN.md) | [Environment and CLI tutorial](../tutorial/01-environment.md) | [Project README](../../README.md)

The installed command is `panoptic-segment`. Run `panoptic-segment <command> --help` for argparse-level details. YAML values passed to `--set` use `KEY=VALUE` and are parsed by PyYAML, so `null`, booleans, numbers, and lists retain their types.

## `show-config`

Print defaults merged with an optional YAML file and CLI overrides. It does not access data.

```bash
uv run panoptic-segment show-config --config configs/learning_minimal.yaml
uv run panoptic-segment show-config --set data.image_size='[256,512]' --set train.amp=false
```

## `prepare-data`

Pair `images/`, `semantic/`, and `instance/` by exact stem and write `train.csv`, `valid.csv`, `test.csv`, `schema.yaml`, and `dataset.yaml`.

```bash
uv run panoptic-segment prepare-data \
  --data-dir data/raw --manifest-dir data/manifests \
  --schema configs/synthetic_schema.yaml --ratios 0.8 0.1 0.1 --seed 42
```

For video, scene, or crop-derived data, pass `--group-file` with `sample_id,group_id` to keep related samples in one split:

```bash
uv run panoptic-segment prepare-data \
  --data-dir data/raw --manifest-dir data/manifests \
  --group-file data/raw/groups.csv \
  --schema configs/synthetic_schema.yaml
```

The generic command creates a random split. Do not use it for a benchmark with an official split; use its converter instead.

## `inspect-data`

Validate manifest metadata and panoptic labels before training.

```bash
uv run panoptic-segment inspect-data --manifest-dir data/manifests
uv run panoptic-segment inspect-data --manifest-dir data/manifests --limit-per-split 8
```

`--limit-per-split` limits decoded content checks. Metadata, manifest hashes, split counts, and cross-split IDs are still checked, but a passing limited inspection is not a full sample validation.

## `convert-cityscapes`

Convert licensed Cityscapes `labelIds` and `instanceIds` while preserving official train/val membership. The command writes the generic masks plus official-format panoptic artifacts.

```bash
uv run panoptic-segment convert-cityscapes \
  --data-root /path/to/cityscapes --output-root data/cityscapes
```

Use `--symlink-images` to avoid copying RGB files on the same filesystem. Use `--non-strict` only while investigating malformed source labels.

## `train`

Start a run, perform a real one-batch update with `--dry-run`, or resume the configured run's `last.pt`.

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml --device cpu --dry-run
uv run panoptic-segment train --config configs/learning_minimal.yaml --set run_name=first-run
uv run panoptic-segment train --config configs/learning_minimal.yaml \
  --set train.epochs=4 --resume artifacts/first-run/last.pt
```

Normal training refuses to overwrite an existing run. Resume permits increasing `train.epochs` and operational path/worker/device changes, but rejects changed model, data semantics, schema, loss, optimizer, scheduler, augmentation, or post-processing.

## `evaluate`

Reload a checkpoint, validate the prepared dataset identity, and evaluate one split. The default is `valid`.

```bash
uv run panoptic-segment evaluate artifacts/first-run/best.pt --split valid --device cpu
uv run panoptic-segment evaluate artifacts/first-run/best.pt --split test \
  --output artifacts/first-run/evaluation.json
```

The optional JSON report records the checkpoint path and SHA-256, split, device, data identity, aggregate metrics, one row per evaluated sample, and the lowest-PQ `worst_cases` list. Use `--worst-cases 0` to omit that shortlist. Evaluation uses the post-processing settings embedded in the checkpoint.

## `predict`

Run saved-size inference for one RGB image and restore discrete masks to the source dimensions.

```bash
uv run panoptic-segment predict artifacts/first-run/best.pt \
  data/raw/images/sample_0000.png --output artifacts/prediction --device cpu
```

The output directory receives semantic IDs, 16-bit instance IDs, schema colors, and an overlay. Cityscapes official-format export uses `scripts/predict_cityscapes.py` instead.

## Related scripts

- `scripts/preview_panoptic.py`: create a source/semantic/panoptic contact sheet.
- `scripts/convert_kaggle_soccer.py`: convert the public Kaggle Soccer dataset.
- `scripts/convert_cityscapes.py`: licensed Cityscapes conversion.
- `scripts/predict_cityscapes.py`: Cityscapes panoptic prediction export.
- `scripts/evaluate_cityscapes.py`: crowd-aware official evaluation wrapper.
- `scripts/evaluate_panopticapi.py`: optional COCO-panoptic-compatible evaluator.
- `scripts/kaggle_train.py`: synthetic GPU reference runner.

Script-specific dependencies and benchmark boundaries are documented in [Scripts](../../scripts/README.md) and the [Cityscapes guide](../guides/cityscapes.md).
