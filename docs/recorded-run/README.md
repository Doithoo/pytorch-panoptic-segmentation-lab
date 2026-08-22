# Kaggle Reference Run Status

[简体中文](README.zh-CN.md) | [Kaggle guide](../guides/kaggle.md) | [Reference config](../../configs/reference_kaggle.yaml)

## Status: pending execution

The submit-ready kernel and artifact contract are complete, but no successful Kaggle GPU output has been retrieved into this repository yet. This page intentionally contains no invented metric, runtime, hardware allocation, or notebook URL.

The pending reference protocol is:

| Item | Fixed value |
|---|---|
| Data | 256 deterministic synthetic images, 128x128 source |
| Split | deterministic 0.8 / 0.1 / 0.1, seed 42 |
| Model | Panoptic U-Net, base channels 32 |
| Training | 20 epochs, AdamW, cosine schedule, CUDA AMP |
| Input | resized to 256x256, batch 4, workers 2 |
| Selection | best validation PQ |
| Final evaluation | test split loaded from `best.pt` |
| Hardware | Kaggle T4 or newer |

Submitted files are `kaggle/run_kaggle.py` and `kaggle/kernel-metadata.json`. The runner records its resolved source commit and final checkpoint hash.

## Completion checklist

A result may replace this status only when:

- Kaggle status is `COMPLETE`;
- all 20 metric rows are present and finite;
- `best.pt` safely reloads for test evaluation;
- summary, resolved config, run metadata, metrics, and evaluation are retained;
- source revision, checkpoint SHA-256, GPU, versions, runtime, and split counts are stated;
- a Kaggle page URL is linked;
- synthetic results are labeled workflow evidence rather than benchmark quality.

Do not commit the large checkpoint. Retain small CSV/YAML/JSON evidence and selected visualizations. A separate real-data record must document converter, license, official split, crowd/void evaluator compatibility, and dataset identity.
