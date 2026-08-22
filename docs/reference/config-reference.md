# Configuration Reference

[简体中文](config-reference.zh-CN.md) | [Configuration flow](../concepts/configuration-flow.md)

All fields are strict. Defaults below are package defaults; experiment YAML may override them.

## `data`

| Field | Default | Contract |
|---|---:|---|
| `data_dir` | `data/raw` | Source location used during preparation/orchestration |
| `manifest_dir` | `data/manifests` | Prepared CSV, schema, and dataset metadata |
| `image_size` | `[256,256]` | `[height,width]`, positive and divisible by 16 |
| `batch_size` | `4` | positive |
| `num_workers` | `0` | nonnegative; operational and may change on resume |
| `horizontal_flip` | `0.5` | probability from 0 to 1 |
| `center_sigma` | `8.0` | positive Gaussian sigma in resized pixels |
| `max_train_samples` | `256` | positive or null |
| `max_valid_samples` | `64` | positive or null |
| `max_test_samples` | null | positive or null; independent from validation |

## `model` and `loss`

| Field | Default | Contract |
|---|---:|---|
| `model.name` | `panoptic_unet_small` | registered model name |
| `in_channels` | `3` | image channels |
| `expected_num_classes` | `3` | must equal prepared schema count |
| `base_channels` | `32` | at least 4 |
| `semantic_weight` | `1.0` | nonnegative CE multiplier |
| `center_weight` | `1.0` | nonnegative center focal multiplier |
| `offset_weight` | `0.01` | nonnegative thing L1 multiplier |
| `ignore_index` | `255` | nonnegative and equal to schema ignore |

## `train`

| Field | Default | Contract |
|---|---:|---|
| `epochs` | `20` | positive; may only increase on resume |
| `lr` | `0.001` | positive |
| `weight_decay` | `0.0001` | nonnegative |
| `optimizer` | `adamw` | `adamw` or `sgd` |
| `scheduler` | `cosine` | `none`, `cosine`, or `step` |
| `scheduler_step_size` | `10` | positive, used by step |
| `scheduler_gamma` | `0.1` | `(0,1]`, used by step |
| `amp` | `true` | CUDA AMP only |
| `grad_clip` | `1.0` | nonnegative; zero disables |
| `seed` | `42` | Python, NumPy, torch, loader seed |
| `best_metric` | `pq` | `pq`, `sq`, `rq`, `pq_thing`, `pq_stuff` |

## `postprocess`

| Field | Default | Contract |
|---|---:|---|
| `center_threshold` | `0.2` | probability in `[0,1]` |
| `nms_kernel` | `7` | positive odd integer |
| `top_k_centers` | `200` | positive global per-image bound |
| `instance_area` | `16` | nonnegative minimum thing pixels |
| `stuff_area` | `64` | nonnegative minimum stuff pixels |

Top-level `device` is `auto`, `cpu`, `cuda`, or `mps`; `output_dir` defaults to `artifacts`; `run_name` defaults to `panoptic-unet-small`.

CLI overrides use YAML values:

```bash
--set train.amp=false --set data.max_train_samples=null --set data.image_size='[256,512]'
```
