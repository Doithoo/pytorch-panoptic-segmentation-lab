# Source Code Tour

[简体中文](code-tour.zh-CN.md) | [How one sample moves through the project](how-it-works.md)

Start with `cli.py` to see how a command enters the package. It only parses arguments; the work lives in the modules below.

## Data

- `data/schema.py`: class IDs, colors, thing/stuff flags, and ignore value.
- `data/manifest.py`: file pairing, random or grouped splits, hashes, and data fingerprint.
- `data/inspection.py`: opens masks and checks their values and relationships.
- `data/registry.py`: named dataset converter functions.
- `data/dataset.py`: loads one manifest row and builds a batch.
- `data/transforms.py`: resizes and flips the image and masks together, then builds targets.
- `data/soccer.py`: converts the public Soccer video annotations.

## Model and training

- `models/__init__.py`: model registry and factory lookup.
- `models/panoptic_unet.py`: encoder, decoder, and semantic/center/offset heads.
- `training/losses.py`: semantic cross-entropy, center focal loss, and thing-only offset L1.
- `training/train.py`: loaders, optimizer, scheduler, training loop, validation, and resume.
- `training/checkpoint.py`: checkpoint fields, safe loading, RNG state, and run metadata.

## Decode and results

- `inference/postprocess.py`: center selection and same-class instance assignment.
- `evaluation/metrics.py`: per-class PQ/SQ/RQ accumulation.
- `evaluation/evaluate.py`: checkpoint loading, identity checks, aggregate metrics, and per-image reports.
- `evaluation/visualization.py`: semantic colors and instance overlays.
- `inference/predictor.py`: checkpoint-backed prediction and original-size output.

Functions whose names start with `_` are internal helpers. Code outside the package should use the CLI or the documented public functions rather than importing those helpers.
