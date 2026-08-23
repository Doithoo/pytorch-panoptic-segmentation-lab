# Code Tour

[简体中文](code-tour.zh-CN.md) | [How it works](how-it-works.md)

Start at `cli.py`; it owns argument parsing only. `config.py` merges defaults, strict YAML, and typed CLI overrides, then validates cross-field ranges.

Data ownership:

- `data/schema.py`: immutable class and thing/stuff meaning.
- `data/manifest.py`: deterministic pairing, split allocation, hashes, identity.
- `data/inspection.py`: decoded label integrity.
- `data/dataset.py`: row loading and batching.
- `data/transforms.py`: synchronized geometry and training targets.

Model and optimization:

- `models/panoptic_unet.py`: shared encoder/decoder and three heads.
- `training/losses.py`: semantic CE, center focal, thing-only offset L1.
- `training/train.py`: loaders, optimizer/scheduler, fit/evaluate, resume.
- `training/checkpoint.py`: safe atomic persistence, RNG, environment, hashes.

Result ownership:

- `inference/postprocess.py`: bounded center extraction and panoptic decode.
- `evaluation/metrics.py`: split-level per-class PQ statistics.
- `evaluation/evaluate.py`: checkpoint-backed split evaluation.
- `evaluation/visualization.py`: schema colors and instance overlays.
- `inference/predictor.py`: saved-size preprocessing, model reload, exact-size exports.

Private helpers prefixed `_` are shared internally where necessary but are not a stable public API. Extensions should add explicit contracts rather than importing deeper private functions from application code.
