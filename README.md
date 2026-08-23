# PyTorch Panoptic Segmentation Lab

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Doithoo/pytorch-panoptic-segmentation-lab/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)
[简体中文](README.zh-CN.md) | [Documentation](docs/README.md) | [Kaggle guide](docs/guides/kaggle.md) | [Cityscapes guide](docs/guides/cityscapes.md)

A readable, reproducible PyTorch project for learning panoptic segmentation end to end. The baseline predicts semantic classes, thing centers, and per-pixel center offsets, then combines them into thing instances and stuff regions.

![Synthetic source, semantic labels, and panoptic overlays](docs/assets/synthetic-panoptic-preview.png)

> Project status: the local and packaged workflows are tested, and the deterministic synthetic Kaggle reference job completed successfully. See [recorded run](docs/recorded-run/README.md) for metrics and evidence. The result is workflow evidence rather than a real-data benchmark; the built-in PQ evaluator covers this project's non-crowd mask contract and is not a replacement for dataset-specific crowd handling or an official benchmark server.

## What is included

- Panoptic U-Net with semantic, center-heatmap, and offset heads.
- Synchronized transforms and Gaussian center targets.
- Thing-only offset supervision and focal center loss.
- Bounded, class-consistent center assignment with configurable area filtering.
- Class-wise PQ/SQ/RQ accumulation, void handling, and thing/stuff summaries.
- Deterministic manifests, dataset identity, and panoptic-label preflight.
- Safe `weights_only=True`, atomic, versioned checkpoints with resume state.
- Resolved configuration, metrics history, environment metadata, and hashes.
- Raw semantic/instance outputs, semantic colors, and panoptic overlays.
- CPU tests and a Kaggle T4 reference runner with heartbeats and final evaluation.
- An official-split Cityscapes converter with raw/train ID validation and panoptic JSON/PNG export.

## Who this is for

This project is for readers who know basic Python, tensors, loss, and gradients but have not completed a panoptic segmentation workflow. The shortest path is synthetic and CPU-friendly; Cityscapes and Kaggle are separate protocol exercises, not prerequisites.

## Quick start

```bash
uv sync --locked --extra dev
uv run python scripts/create_synthetic_data.py
uv run panoptic-segment prepare-data --schema configs/synthetic_schema.yaml
uv run panoptic-segment inspect-data
uv run python scripts/preview_panoptic.py data/manifests/train.csv \
  --output artifacts/dataset-preview.png --limit 4
uv run panoptic-segment train --config configs/learning_minimal.yaml --dry-run
uv run panoptic-segment train --config configs/learning_minimal.yaml
```

Evaluate and predict from the validation-selected checkpoint:

```bash
uv run panoptic-segment evaluate artifacts/learning-minimal/best.pt --split test \
  --output artifacts/learning-minimal/evaluation.json
uv run panoptic-segment predict artifacts/learning-minimal/best.pt \
  data/raw/images/sample_0000.png --output artifacts/prediction
```

The optional JSON report includes the checkpoint hash, data identity, aggregate metrics, one row per evaluated sample, and the lowest-PQ failure shortlist.

A prediction writes:

```text
sample_0000.semantic.png
sample_0000.instance.png
sample_0000.semantic-color.png
sample_0000.overlay.png
```

## Data contract

Each sample has three files with the same stem:

```text
data/raw/
  images/sample_0001.png
  semantic/sample_0001.png   # contiguous class ID or 255
  instance/sample_0001.png   # 0 for stuff/void, positive for things
```

Every positive instance ID must belong to exactly one thing class in one image. Thing pixels require a positive instance ID; stuff and ignored pixels require zero. `prepare-data` rejects unmatched stems and writes portable relative-path manifests. `inspect-data` validates decoded images, dimensions, labels, instances, split counts, and cross-split IDs.

See [data format](docs/reference/data-format.md) before adapting a dataset.

## Training artifacts

A normal run writes `artifacts/<run_name>/`:

| File | Purpose |
|---|---|
| `config.yaml` | Fully resolved settings used by the run |
| `run.yaml` | Python/PyTorch/platform/device, Git revision, data identity, times |
| `metrics.csv` | Loss components, learning rate, PQ/SQ/RQ, thing/stuff PQ |
| `last.pt` | Latest resumable model, optimizer, scheduler, scaler, RNG, history |
| `best.pt` | Checkpoint selected by `train.best_metric` |

Resume only a compatible run:

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml \
  --set train.epochs=4 --resume artifacts/learning-minimal/last.pt
```

Checkpoint loading is safe and schema-versioned. Do not bypass the project loader with `weights_only=False` for untrusted files.

## Configuration

YAML is strict: unknown fields fail instead of being ignored. Selected values can be overridden from the CLI:

```bash
uv run panoptic-segment show-config --config configs/learning_minimal.yaml \
  --set data.batch_size=4 --set run_name=experiment-01
```

Training input dimensions must be divisible by 16 because the model downsamples four times. Source images may have other dimensions; the training transform resizes them to `data.image_size`, and prediction restores discrete outputs to the original size.

See the [configuration reference](docs/reference/config-reference.md).

## Cityscapes

Cityscapes support preserves its official train/val split and converts `labelIds` plus `instanceIds` into the project contract:

```bash
uv run panoptic-segment convert-cityscapes \
  --data-root /path/to/cityscapes --output-root data/cityscapes
uv run panoptic-segment inspect-data --manifest-dir data/cityscapes
```

Read the [Cityscapes guide](docs/guides/cityscapes.md) before training. Public Cityscapes test annotations are unavailable, so the project reports validation unless an official server is used. The converter also writes official-format panoptic JSON/PNG artifacts for optional evaluator integration.

## Kaggle GPU

The repository includes a no-dataset synthetic reference kernel. It is intended to prove that source retrieval, CUDA kernels, training, checkpoint reload, test evaluation, and artifact export all work in one non-interactive Kaggle job.

```bash
uv tool install kaggle
kaggle auth login
# Edit your account in docs/recorded-run/kaggle/kernel-metadata.json
kaggle kernels push -p docs/recorded-run/kaggle
```

Use a T4 or newer NVIDIA GPU. The runner records the resolved Git commit and checkpoint SHA-256. Publishing a real Cityscapes or COCO result additionally requires a dataset converter, official split policy, dataset-specific crowd/void behavior, and compliance with that dataset's license.

The repository also documents a public, small-scale alternative: the [Kaggle Soccer workflow](docs/guides/kaggle-soccer.md) converts video/COCO polygon annotations into the project's three-mask contract before training. It is a teaching protocol, not an official benchmark.

Read the [complete Kaggle guide](docs/guides/kaggle.md).

## Learning path

1. [Tensors and panoptic IDs](docs/tutorial/00-basics.md)
2. [Environment and CLI](docs/tutorial/01-environment.md)
3. [Data, center heatmaps, and offsets](docs/tutorial/02-data-and-targets.md)
4. [Panoptic U-Net](docs/tutorial/03-panoptic-unet.md)
5. [Training, artifacts, and resume](docs/tutorial/04-training.md)
6. [Evaluation, prediction, and limitations](docs/tutorial/05-evaluation-and-inference.md)

Start with the [guided learning path](docs/tutorial/learning-path.md) or use the [code tour](docs/concepts/code-tour.md) when reading the implementation.

## Repository map

```text
configs/                     teaching, reference, and Cityscapes settings
docs/tutorial/               guided chapters and runnable learning path
docs/guides/                 task procedures and benchmark boundaries
docs/reference/              exact data, metric, CLI, and checkpoint contracts
examples/                    small programs for targets, heads, and workflow
scripts/                     dataset, evaluator, and Kaggle orchestration
src/panoptic_segmenter/      typed installed package
tests/                       offline unit, integration, and documentation tests
```

The [documentation index](docs/README.md) is organized by intent. Start with the [tutorial index](docs/tutorial/README.md) or go directly to the [CLI reference](docs/reference/cli.md).

## Scope and limitations

The baseline is intentionally small and trained from scratch. It demonstrates the complete contract but does not claim Panoptic-DeepLab architectural parity or state-of-the-art accuracy. The project currently has no pretrained backbone, distributed training, or completed real-dataset Kaggle record. Cityscapes conversion and official-format export are included; an official leaderboard result still requires the benchmark evaluator and policy. These are explicit extension points, not hidden claims.

Run all local quality gates with:

```bash
make check
```

Contributions should preserve readable contracts, include hand-checkable metric cases, and document any new checkpoint or target fields. See [CONTRIBUTING.md](CONTRIBUTING.md).
