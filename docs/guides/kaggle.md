# Complete Training on Kaggle

[简体中文](kaggle.zh-CN.md) | [Recorded-run status](../recorded-run/README.md)

The supplied kernel completes a deterministic 256-image synthetic run without an attached Kaggle Dataset. For a small public-data teaching workflow, use the [Kaggle Soccer guide](kaggle-soccer.md), which converts `quantigoai/soccer-dataset` before training. The synthetic job validates the non-interactive GPU and artifact workflow; neither workflow is an official benchmark.

## Prerequisites

Install the Kaggle CLI separately and authenticate:

```bash
uv tool install kaggle
kaggle auth login
kaggle --version
```

Push the repository changes to GitHub before submission because the kernel clones the repository. Open `docs/recorded-run/kaggle/kernel-metadata.json`, replace `your-username`, and keep GPU plus Internet enabled.

## Submit and monitor

```bash
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <username>/pytorch-panoptic-segmentation-lab-gpu
```

The runner clones the repository, checks out `main` unless `PANOPTIC_REVISION` is set, records the resolved commit, installs the package, generates data, runs the CUDA preflight, trains, reloads `best.pt`, evaluates test, and writes a summary. For a permanent reference run, replace `REVISION` in `run_kaggle.py` with the pushed commit SHA before submission.

Use a T4 or newer NVIDIA GPU. Do not select P100: current Kaggle PyTorch builds may not contain kernels for its `sm_60` compute capability. The runner performs a CUDA forward/backward operation, not only `torch.cuda.is_available()`.

A healthy log contains JSON phases for `source`, `preflight`, per-epoch lines, 60-second `training` heartbeats, and `complete`. A heartbeat means the process is alive; it does not mean an epoch has finished.

## Download artifacts

After status becomes `COMPLETE`:

```bash
kaggle kernels output <username>/pytorch-panoptic-segmentation-lab-gpu \
  --file-pattern 'artifacts/.*|kaggle-run-summary.json' -p kaggle-output
```

Inspect:

| File | Check |
|---|---|
| `kaggle-run-summary.json` | complete status, GPU, source revision, duration, split counts, test metrics, checkpoint hash |
| `artifacts/reference-panoptic-unet/config.yaml` | resolved CUDA/AMP/data/postprocess settings |
| `run.yaml` | environment, data identity, Git revision, timings |
| `metrics.csv` | 20 rows, finite component losses, validation metrics |
| `best.pt` / `last.pt` | validation-selected versus final resumable state |
| `evaluation/evaluation.json` | automatic test summary from `best.pt` |
| `evaluation/evaluation_detailed.json` | checkpoint/data identity, per-image metrics, and lowest-PQ cases |
| `evaluation/per_class.csv` | PQ/SQ/RQ row for every schema class |

Copy small evidence into `docs/recorded-run/`, update both recorded-run READMEs, and link the Kaggle page. Do not commit large checkpoints.

## Run with your dataset

For the generic three-folder contract, attach one private Kaggle Dataset and call `scripts/kaggle_train.py --input /kaggle/input/<dataset>`. Supply the matching schema with `--schema` and a config whose class count and ignore ID agree.

Cityscapes and COCO source formats do not match the generic contract directly. A credible real-data result requires:

1. a documented converter and mapping to contiguous IDs;
2. official train/validation membership rather than random splitting;
3. dataset-specific crowd/void semantics and official evaluator comparison;
4. source identity and license-compliant distribution;
5. per-class and visual error artifacts.

Cityscapes test labels are not public, so report validation or use its official server. Never relabel validation as test.

## Common failures

- `repository not found`: push the repository or correct `REPOSITORY`.
- checkout failure: pin a commit that exists on the remote.
- no CUDA: enable GPU and restart the kernel.
- CUDA kernel failure on P100: choose T4 or newer.
- OOM: lower batch size first, then image size or model width; center top-k already bounds decode memory.
- resume mismatch: use the same prepared data/schema/config and only increase epochs.
- `dataset stems do not match`: repair all three input folders before training.

The first published synthetic result proves reproducibility mechanics. Publish a real benchmark only after the additional protocol work above is complete.
