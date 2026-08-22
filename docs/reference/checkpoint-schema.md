# Checkpoint Schema Version 1

[简体中文](checkpoint-schema.zh-CN.md) | [Training tutorial](../tutorial/04-training.md)

Only schema version 1 is accepted. `save_checkpoint` writes a unique temporary file and publishes it with `os.replace`. `load_checkpoint` calls `torch.load(..., weights_only=True)` and rejects missing/version-incompatible mappings.

## Top-level fields

| Field | Meaning |
|---|---|
| `schema_version` | integer `1` |
| `config` | complete resolved experiment configuration |
| `schema` | ordered classes, thing flags, colors, ignore ID |
| `model_state` | tensor state dict |
| `optimizer_state` | optimizer groups and tensors |
| `scheduler_state` | scheduler mapping or null |
| `scaler_state` | CUDA GradScaler mapping, possibly empty |
| `epoch` | last completed epoch |
| `best_metric` | best configured validation value through this epoch |
| `metrics` | complete list of CSV row mappings |
| `run_metadata` | Python, torch, torchvision, platform, device, CUDA, seed, Git revision |
| `dataset_identity` | prepared schema/manifest identity |
| `rng_state` | Python, NumPy, torch, and available CUDA states |

Prediction needs config, schema, and model state. Resume additionally needs optimizer, scheduler, scaler, RNG, metrics, and identity.

## Resume compatibility

Exact equality is required for schema, dataset identity, model, loss, post-processing, optimizer/scheduler/seed/AMP/gradient settings, and data semantics such as image size, batch size, augmentation, target sigma, and sample limits. `train.epochs` may increase. Data/manifest paths, worker count, and device are operational and may change, but resume must use `last.pt` inside the configured run directory so history and the historical best remain coherent.

The destination metrics file is appended. Starting normal training in an existing run directory fails, preventing silent history replacement.

## Trust boundary

Use the project loader:

```python
from panoptic_segmenter.training.checkpoint import load_checkpoint

checkpoint = load_checkpoint("artifacts/run/best.pt")
```

Do not fall back to `weights_only=False` for an untrusted checkpoint. Architecture reconstruction uses the saved built-in model name; future external factories would require a separately documented code-trust boundary and checkpoint schema decision.
