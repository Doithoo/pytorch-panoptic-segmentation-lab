# Scripts

[简体中文](README.zh-CN.md) | [CLI reference](../docs/reference/cli.md) | [Cityscapes guide](../docs/guides/cityscapes.md)

Scripts orchestrate package APIs. Training semantics belong in `src/panoptic_segmenter`, so scripts remain thin, testable entry points rather than a second implementation.

| Script | Purpose | Extra requirements |
|---|---|---|
| `create_synthetic_data.py` | Generate deterministic teaching samples | package dependencies |
| `convert_kaggle_soccer.py` | Convert the public Kaggle Soccer video/COCO annotations | `opencv-python-headless` |
| `preview_panoptic.py` | Create source/semantic/panoptic contact sheets | package dependencies |
| `convert_cityscapes.py` | Convert licensed Cityscapes train/val data | Cityscapes archive |
| `predict_cityscapes.py` | Export checkpoint predictions as Cityscapes panoptic PNG/JSON | converted Cityscapes data |
| `evaluate_cityscapes.py` | Generate crowd-aware GT or run official validation | `cityscapesscripts` via `uv run --with` |
| `evaluate_panopticapi.py` | Run a generic panoptic evaluator | `panopticapi` in a separate environment |
| `kaggle_train.py` | Run the synthetic CUDA reference workflow | Kaggle GPU and Internet |
| `kaggle_cityscapes.py` | Run licensed private-data Cityscapes workflow | private Kaggle dataset, GPU, license |

## Local examples

```bash
uv run python scripts/create_synthetic_data.py --count 24 --size 128
uv run --with opencv-python-headless python scripts/convert_kaggle_soccer.py --help
uv run python scripts/preview_panoptic.py data/manifests/train.csv \
  --output artifacts/dataset-preview.png --limit 4
uv run python scripts/convert_cityscapes.py --help
uv run python scripts/predict_cityscapes.py --help
uv run python scripts/evaluate_cityscapes.py --help
```

The generic three-folder workflow should use the installed CLI. The Cityscapes scripts preserve official split membership and provider IDs; do not replace them with a random `prepare-data` call.

## Official evaluation

Install optional evaluator packages only for the benchmark command being run. Keep their versions and policies in the recorded result. An internal non-crowd PQ result is not an official Cityscapes or COCO score.

```bash
uv run --with cityscapesscripts python scripts/evaluate_cityscapes.py --help
uv run python scripts/evaluate_panopticapi.py --help
```

Never commit datasets, credentials, Kaggle tokens, checkpoints, or generated run directories. See [SECURITY.md](../SECURITY.md) for trust boundaries.
