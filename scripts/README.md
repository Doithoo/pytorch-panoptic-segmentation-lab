# Scripts

[简体中文](README.zh-CN.md) | [Kaggle guide](../docs/guides/kaggle.md)

- `create_synthetic_data.py`: generate deterministic-shaped teaching samples; supports `--count`, `--size`, and `--output`.
- `kaggle_train.py`: CUDA preflight, prepare, train/resume, test evaluation, and final summary for a Kaggle working directory.

Scripts orchestrate package APIs. Training semantics belong in `src/panoptic_segmenter`, so scripts remain testable and do not become a second implementation.
