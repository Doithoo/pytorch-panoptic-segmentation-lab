# PyTorch Panoptic Segmentation Lab

A reproducible, beginner-oriented panoptic segmentation project. It connects the pixel-level workflow of semantic segmentation with the instance-level evaluation and inference workflow used in object detection, while keeping the implementation independent from unfinished neighboring projects.

## What is included

- A compact Panoptic-DeepLab-style U-Net with semantic, center, and offset heads.
- A dataset contract based on `image + semantic mask + instance mask`.
- Center NMS, offset-based instance assignment, and thing/stuff handling.
- Panoptic Quality (`PQ`), Segmentation Quality (`SQ`), and Recognition Quality (`RQ`).
- CLI commands for data preparation, dry runs, training, evaluation, and prediction.
- A Kaggle GPU entry point with CUDA preflight and artifact export.

## Quick start

```bash
cd pytorch-panoptic-segmentation-lab
uv sync --extra dev
uv run python scripts/create_synthetic_data.py
uv run panoptic-segment prepare-data --schema configs/cityscapes_mini_schema.yaml
uv run panoptic-segment train --config configs/learning_minimal.yaml --dry-run
uv run panoptic-segment train --config configs/learning_minimal.yaml --device cpu
```

Artifacts are written to `artifacts/<run_name>/`: resolved config, metrics, `last.pt`, and `best.pt`.

## Dataset contract

```text
data/raw/
  images/sample_0001.png
  semantic/sample_0001.png   # class id per pixel
  instance/sample_0001.png   # 0 for stuff, 1, 2, ... for thing instances
```

Stems must match. The `isthing` field in the schema controls whether a class is evaluated as separate instances or as one stuff region. Use a 16-bit PNG or integer image mode when instance IDs exceed 255.

## Kaggle GPU

Add a Kaggle dataset whose root contains `images/`, `semantic/`, and `instance/`, enable a free GPU in Notebook settings, then run:

```bash
!pip install -e /kaggle/working/pytorch-panoptic-segmentation-lab
!python scripts/kaggle_train.py --input /kaggle/input/your-panoptic-dataset
```

The script validates CUDA, prints the allocated GPU, creates manifests under `/kaggle/working/manifests`, and writes checkpoints and metrics under `/kaggle/working/artifacts`. Start with `learning_minimal.yaml` on free-tier hardware.

See `docs/guides/kaggle.md` and `docs/reference/data-format.md` for details.
