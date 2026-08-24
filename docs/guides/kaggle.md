# Complete a Kaggle GPU Run

[简体中文](kaggle.zh-CN.md) | [Kaggle Soccer](kaggle-soccer.md) | [Recorded runs](../recorded-run/README.md)

The repository contains two Kaggle paths:

- the synthetic path below, which checks that the package, CUDA, training loop, checkpoint reload, and artifact export work together;
- [Kaggle Soccer](kaggle-soccer.md), which starts from a public video dataset and exercises annotation conversion before training.

Neither path is an official Cityscapes or COCO benchmark.

## Before submitting

Install and sign in to the Kaggle CLI:

```bash
uv tool install kaggle
kaggle auth login
kaggle --version
```

Push the repository revision that the kernel should run. Open `docs/recorded-run/kaggle/kernel-metadata.json`, set your Kaggle username, and keep GPU and Internet enabled.

## Submit and monitor

```bash
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <username>/pytorch-panoptic-segmentation-lab-gpu
```

The kernel clones the repository, checks out its pinned revision, installs the package, creates data, checks CUDA, trains, reloads `best.pt`, evaluates test, and writes a summary. Pin a full commit before keeping a result as a reference.

Use a T4 or newer GPU. Do not select P100 for current Kaggle PyTorch images; the CUDA build may not contain kernels for `sm_60`. The runner tests an actual CUDA forward and backward pass.

The log includes source revision, preflight, epoch lines, periodic heartbeats, and completion. A heartbeat only means the process is still alive.

## Download the files

After the status is `COMPLETE`:

```bash
kaggle kernels output <username>/pytorch-panoptic-segmentation-lab-gpu \
  --file-pattern 'artifacts/.*|kaggle-run-summary.json' -p kaggle-output
```

Inspect:

| File | What to check |
|---|---|
| `kaggle-run-summary.json` | device, source revision, duration, split counts, test metrics, checkpoint hash |
| `config.yaml` | final data, model, loss, and post-processing values |
| `run.yaml` | software versions, data fingerprint, Git revision, and timing |
| `metrics.csv` | one row per epoch and validation metrics |
| `best.pt` / `last.pt` | selected checkpoint and latest resumable checkpoint |
| `evaluation/evaluation.json` | aggregate test metrics |
| `evaluation/evaluation_detailed.json` | per-image metrics and lowest-PQ samples |
| `evaluation/per_class.csv` | PQ/SQ/RQ for each class |

Copy only small result files into `docs/recorded-run/`. Keep checkpoints and datasets in Kaggle output.

## Use another dataset

If the data already follows `images/`, `semantic/`, and `instance/`, attach it and run:

```bash
python scripts/kaggle_train.py \
  --input /kaggle/input/<dataset> \
  --schema configs/my_schema.yaml \
  --config configs/my_config.yaml
```

If the source uses videos, COCO JSON, Cityscapes IDs, or another annotation format, write the converter first. Keep the source license, class mapping, split rule, and evaluator separate from the training code.

## Common failures

- Repository not found: push the repository or correct `REPOSITORY`.
- Checkout failed: pin a commit that exists on GitHub.
- CUDA unavailable: enable GPU and restart the kernel.
- P100 kernel failure: choose a T4 or newer GPU.
- Out of memory: lower batch size, image size, or `base_channels`.
- Resume mismatch: use the same data, schema, and settings; only increase epochs.
- Stem mismatch: repair the three input directories before training.

For public data with a real conversion step, use the [Kaggle Soccer guide](kaggle-soccer.md).
