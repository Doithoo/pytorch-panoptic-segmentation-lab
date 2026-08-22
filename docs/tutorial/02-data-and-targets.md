# 02: Data and Supervision Targets

[简体中文](02-data-and-targets.zh-CN.md) | [Previous](01-environment.md) | [Next](03-panoptic-unet.md)

Generate, prepare, and inspect before training:

```bash
uv run python scripts/create_synthetic_data.py --count 24 --size 128
uv run panoptic-segment prepare-data --schema configs/synthetic_schema.yaml
uv run panoptic-segment inspect-data
```

Preparation requires exact stem equality across `images`, `semantic`, and `instance`. It rejects duplicate stems, missing directories, invalid ratios, and a sample count too small for non-empty requested splits. A fixed seed shuffles IDs; each nonzero split receives at least one sample. Manifests store paths relative to their directory.

`dataset.yaml` records format version, relative source root, seed, ratios, split counts, manifest hashes, schema hash, and a preparation identity. This identity binds resume to the prepared protocol; it does not claim to hash every source image byte.

Preflight decodes every prepared sample and checks:

- image/semantic/instance dimensions match;
- semantic IDs are declared or ignore;
- instance IDs are nonnegative;
- ignore and stuff pixels use instance zero;
- thing pixels use a positive ID;
- one positive instance does not span semantic classes;
- manifest counts and cross-split sample IDs are consistent.

Images resize with bilinear interpolation; masks use nearest neighbor. Geometry is synchronized before targets are built, so offset units belong to the resized training grid. A horizontal flip updates all three source images together.

Each valid thing instance produces a Gaussian centered at its pixel-coordinate mean. Overlapping Gaussians use their maximum. Offset loss is enabled only on validated thing pixels. Ignore pixels are excluded from center and semantic supervision.

The generic random split is suitable for independent custom samples and synthetic learning data. Real benchmarks must preserve their official split and scene/group boundaries; do not randomly split video frames or neighboring crops.
