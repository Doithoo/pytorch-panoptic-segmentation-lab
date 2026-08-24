# Kaggle Soccer Run

[简体中文](README.zh-CN.md) | [Public Kaggle guide](../../guides/kaggle-soccer.md) | [Kernel page](https://www.kaggle.com/code/yashowhoo/pytorch-panoptic-segmentation-lab-kaggle-soccer)

## Status: complete

Kaggle kernel version 2 completed a full public-data teaching run: source checkout, Soccer video/COCO polygon conversion, group-aware manifest preparation, CUDA preflight, training, checkpoint reload, test evaluation, and detailed failure metrics.

This run shows a complete public-data path: source checkout, annotation conversion, grouped split, CUDA training, checkpoint reload, test evaluation, and per-image error output. It is not an official benchmark. The source dataset is CC-BY-SA-4.0; frames are sampled with `frame_stride=5` and split by source video.

## Result

| Item | Value |
|---|---:|
| Kaggle version | 2 |
| Hardware | Tesla T4 |
| Python / PyTorch | 3.12.13 / 2.10.0+cu128 |
| Source revision | `e88a11d488ad0f02f476f7143c76484b73ed579b` |
| Conversion | 240 max frames, stride 5, width 512 |
| Split | 85 train / 46 valid / 39 test |
| Groups | train `Batch 3`, valid `Batch 1`, test `Batch 2` |
| Model | Panoptic U-Net, base channels 16 |
| Training | 10 epochs, AdamW, cosine schedule, CUDA AMP |
| Best validation PQ | **0.290397** at epoch 6 |
| Test PQ | **0.223444** |
| Test SQ / RQ | 0.878695 / 0.251701 |
| Test PQ thing / stuff | **0.000000 / 0.391027** |
| Test TP / FP / FN | 63 / 5618 / 266 |
| Total elapsed | 149.5 seconds |
| Best checkpoint SHA-256 | `3e0ea31a7f1482702752beea285f6fdd8c27b5bb7468f35e92a582e3ca4f2d08` |

The low thing score and high false-positive count show a concrete failure: this small from-scratch model learns broad stuff regions but does not separate soccer players, balls, and referees. Use the center/offset settings, post-processing values, and per-image report to investigate that failure.

## Evidence files

| File | Contents |
|---|---|
| [`kaggle-run-summary.json`](kaggle-run-summary.json) | GPU, source revision, conversion, split counts, metrics, checkpoint hash |
| [`reference-kaggle-soccer/config.yaml`](reference-kaggle-soccer/config.yaml) | Resolved seven-class training configuration |
| [`reference-kaggle-soccer/dataset.yaml`](reference-kaggle-soccer/dataset.yaml) | Data identity and group-aware split proof |
| [`reference-kaggle-soccer/run.yaml`](reference-kaggle-soccer/run.yaml) | Environment, device, seed, and timing metadata |
| [`reference-kaggle-soccer/metrics.csv`](reference-kaggle-soccer/metrics.csv) | All ten training/validation rows |
| [`reference-kaggle-soccer/evaluation/evaluation.json`](reference-kaggle-soccer/evaluation/evaluation.json) | Aggregate test metrics |
| [`reference-kaggle-soccer/evaluation/evaluation_detailed.json`](reference-kaggle-soccer/evaluation/evaluation_detailed.json) | Per-image metrics and worst cases |
| [`reference-kaggle-soccer/evaluation/per_class.csv`](reference-kaggle-soccer/evaluation/per_class.csv) | Per-class PQ/SQ/RQ |

The checkpoint is intentionally not committed. Download it from the Kaggle output when needed:

```bash
kaggle kernels output yashowhoo/pytorch-panoptic-segmentation-lab-kaggle-soccer \
  --file-pattern 'artifacts/kaggle-soccer-panoptic-unet/best.pt' -p kaggle-soccer-output
```

## Reproduce

The kernel files are [`run_kaggle.py`](run_kaggle.py) and [`kernel-metadata.json`](kernel-metadata.json). The runner pins the reviewed source revision and attaches `quantigoai/soccer-dataset`.

```bash
kaggle kernels push -p docs/recorded-run/kaggle-soccer
kaggle kernels status yashowhoo/pytorch-panoptic-segmentation-lab-kaggle-soccer
```

## Limitations

The run uses a small public video-derived dataset, not Cityscapes or COCO. Group-aware splitting prevents adjacent-video leakage, but the source dataset has no official benchmark protocol or project-specific crowd policy. The result must not be compared directly with leaderboard scores.
