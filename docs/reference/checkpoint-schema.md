# Checkpoint Schema v1

[简体中文](checkpoint-schema.zh-CN.md) | [Training](../tutorial/04-training.md)

Only schema version 1 is accepted. Saving writes a temporary file and then replaces the destination with `os.replace`. Loading uses `torch.load(..., weights_only=True)` and rejects missing or incompatible fields.

## Fields

| Field | Meaning |
|---|---|
| `schema_version` | integer `1` |
| `config` | final experiment settings |
| `schema` | ordered classes, thing flags, colors, ignore ID |
| `model_state` | tensor state dict |
| `optimizer_state` | optimizer state |
| `scheduler_state` | scheduler state or null |
| `scaler_state` | CUDA scaler state, possibly empty |
| `epoch` | last completed epoch |
| `best_metric` | best validation value so far |
| `metrics` | rows written to `metrics.csv` |
| `run_metadata` | Python, torch, torchvision, platform, device, seed, Git revision |
| `dataset_identity` | fingerprint of the prepared schema and manifests |
| `rng_state` | Python, NumPy, torch, and CUDA random states when available |

Prediction needs `config`, `schema`, and `model_state`. Resume also needs the optimizer, scheduler, scaler, RNG, history, and data identity.

## Resume rules

Resume requires the same schema, data fingerprint, model, loss, post-processing, optimizer, scheduler, seed, augmentation, target sigma, image size, batch size, and sample limits. You may increase `train.epochs`. Data paths, worker count, and device are operational settings that may change.

Resume must use `last.pt` inside the configured run directory. A normal train refuses to overwrite an existing metrics file.

## Loading untrusted files

Use the project loader:

```python
from panoptic_segmenter.training.checkpoint import load_checkpoint

checkpoint = load_checkpoint("artifacts/run/best.pt")
```

Do not switch to `weights_only=False` for a checkpoint you did not create or inspect. Rebuilding the model executes the installed model factory named in the saved configuration.
