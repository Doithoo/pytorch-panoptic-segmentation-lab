# 02: Data and Supervision Targets

[简体中文](02-data-and-targets.zh-CN.md) | [Previous](01-environment.md) | [Next](03-panoptic-unet.md)

Start with generated data so you can inspect every label. Run:

```bash
uv run python scripts/create_synthetic_data.py --count 24 --size 128
uv run panoptic-segment prepare-data --schema configs/synthetic_schema.yaml
uv run panoptic-segment inspect-data
uv run python scripts/preview_panoptic.py data/manifests/train.csv \
  --output artifacts/target-preview.png --limit 4
```

The three source folders must contain exactly the same stems. Preparation rejects duplicate stems, missing folders, invalid ratios, and too few samples for the requested splits. A fixed seed controls random splitting. For related video frames or crops, pass `--group-file` instead.

`dataset.yaml` stores the source path, seed, ratios, split counts, file hashes, schema hash, data fingerprint, and, for grouped splits, the group file and groups assigned to each split. It does not hash every source image byte.

`inspect-data` opens each selected sample and checks:

- image, semantic, and instance dimensions;
- declared semantic IDs and the ignore value;
- non-negative instance IDs;
- zero instance IDs on stuff and ignored pixels;
- positive instance IDs on thing pixels;
- one semantic class per positive instance;
- manifest counts, hashes, and split membership;
- group membership when a group file is present.

Images use bilinear interpolation and masks use nearest-neighbor interpolation. The same resize and horizontal flip are applied to all three inputs before targets are built. Offsets therefore use pixels in the resized training grid.

For each valid thing instance, `build_targets` places a Gaussian peak at the mean pixel coordinate and stores `[dy, dx]` for each thing pixel. Center loss ignores the non-center background shoulders; offset loss uses only thing pixels; ignore pixels are excluded from semantic and center supervision.

A random split is fine for independent synthetic samples. Preserve official splits for benchmark data, and split by group for videos, scenes, patients, or crops from one larger image.
