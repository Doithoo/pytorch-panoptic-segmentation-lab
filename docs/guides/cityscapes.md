# Cityscapes Workflow

[简体中文](cityscapes.zh-CN.md) | [Data format](../reference/cityscapes.md) | [Kaggle guide](kaggle.md)

Cityscapes is a licensed dataset. Download it from the official website after accepting its terms; the repository does not redistribute it and does not download it automatically.

## Expected extracted tree

```text
/path/to/cityscapes/
  leftImg8bit/train/<city>/*_leftImg8bit.png
  leftImg8bit/val/<city>/*_leftImg8bit.png
  gtFine/train/<city>/*_gtFine_labelIds.png
  gtFine/train/<city>/*_gtFine_instanceIds.png
  gtFine/val/<city>/*_gtFine_labelIds.png
  gtFine/val/<city>/*_gtFine_instanceIds.png
```

Convert with the official split preserved:

```bash
uv run panoptic-segment convert-cityscapes \
  --data-root /path/to/cityscapes \
  --output-root data/cityscapes
uv run panoptic-segment inspect-data --manifest-dir data/cityscapes
uv run panoptic-segment show-config --config configs/cityscapes.yaml
uv run panoptic-segment train --config configs/cityscapes.yaml \
  --set data.manifest_dir=data/cityscapes \
  --set data.data_dir=data/cityscapes
```

The converter maps official raw label IDs to the 19 contiguous train IDs, decodes `category_id * 1000 + instance_id`, re-indexes instances per image, and writes `train.csv` from Cityscapes train and `valid.csv` from Cityscapes val. `test.csv` is intentionally empty because public Cityscapes test annotations are not available. `inspect-data` allows that explicitly declared unavailable test split.

Use `--symlink-images` when the prepared directory is on the same filesystem and you do not want to copy the large RGB images. Use strict mode by default. `--non-strict` is only for investigating malformed source labels and must not be used for a published result.

## Official-format artifacts

The converter additionally writes:

```text
panoptic_train.json
panoptic_valid.json
panoptic/train/*.png
panoptic/valid/*.png
```

The PNGs use the standard RGB encoding of `category_id * 1000 + instance_id`, and the JSON contains `images`, `annotations`, `segments_info`, and 19 categories. This is the bridge to an official-format evaluator; the project's training evaluator still consumes the three masks.

For an official comparison, install and pin `cityscapesscripts` in a separate benchmark environment. Generate crowd-aware ground truth directly from the original `instanceIds`:

```bash
uv run --with cityscapesscripts python scripts/evaluate_cityscapes.py prepare-ground-truth \
  --cityscapes-root /path/to/cityscapes \
  --output data/cityscapes-official-panoptic --split val
```

Then export predictions from the selected checkpoint:

```bash
uv run python scripts/predict_cityscapes.py artifacts/cityscapes-panoptic-unet/best.pt \
  --manifest data/cityscapes/valid.csv \
  --output artifacts/cityscapes-predictions --device cuda
```

Run the official evaluator:

```bash
uv run --with cityscapesscripts python scripts/evaluate_cityscapes.py evaluate \
  --ground-truth-json data/cityscapes-official-panoptic/cityscapes_panoptic_val.json \
  --prediction-json artifacts/cityscapes-predictions/predictions.json \
  --ground-truth-folder data/cityscapes-official-panoptic/cityscapes_panoptic_val \
  --prediction-folder artifacts/cityscapes-predictions/panoptic \
  --results artifacts/cityscapes-official-results.json
```

The generic `scripts/evaluate_panopticapi.py` wrapper is also available for other COCO-panoptic-compatible protocols, but Cityscapes results should use `cityscapesscripts`. Record the package version and exact policy. A result from the internal PQ evaluator must not be labeled official merely because it has the same acronym.

## Data and evaluation boundaries

- Cityscapes train/val membership is fixed; do not call `prepare-data` on the converted directory because that would randomly reshuffle it.
- Raw IDs 0–33 are not the model class IDs; use the converter's schema.
- Ignore labels map to 255 and instance ID 0.
- `caravan`, `trailer`, and other raw labels with train ID 255 are ignored.
- Cityscapes test labels are private; report validation or use the official server instead of inventing a local test score.
- Keep the original license and source version in the run record.
