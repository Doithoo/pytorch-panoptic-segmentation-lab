# Scripts

[简体中文](README.zh-CN.md) | [Kaggle guide](../docs/guides/kaggle.md)

- `create_synthetic_data.py`: generate deterministic-shaped teaching samples; supports `--count`, `--size`, and `--output`.
- `kaggle_train.py`: CUDA synthetic reference workflow.
- `kaggle_cityscapes.py`: licensed private-dataset Cityscapes conversion, training, prediction, and official validation.
- `convert_cityscapes.py`: convert licensed Cityscapes train/val data with official splits and panoptic JSON/PNG artifacts.
- `evaluate_panopticapi.py`: optional generic panoptic evaluator wrapper.
- `evaluate_cityscapes.py`: generate official crowd-aware GT and run `cityscapesscripts` evaluation.
- `predict_cityscapes.py`: export a checkpoint's validation predictions as panoptic PNG/JSON.

Scripts orchestrate package APIs. Training semantics belong in `src/panoptic_segmenter`, so scripts remain testable and do not become a second implementation.
