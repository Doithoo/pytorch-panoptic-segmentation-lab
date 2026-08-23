# Configuration Reference

[简体中文](config-reference.zh-CN.md) | [Configuration flow](../concepts/configuration-flow.md)

All fields are strict. Defaults below are package defaults; experiment YAML may override them.

## `data`

| Field | Default | Contract |
|---|---:|---|
| `data.data_dir` | `data/raw` | Source location used during preparation/orchestration |
| `data.manifest_dir` | `data/manifests` | Prepared CSV, schema, and dataset metadata |
| `data.image_size` | `[256,256]` | `[height,width]`, positive and divisible by 16; source images are resized to this grid |
| `data.batch_size` | `4` | positive |
| `data.num_workers` | `0` | nonnegative; operational and may change on resume |
| `data.horizontal_flip` | `0.5` | probability from 0 to 1 |
| `data.center_sigma` | `8.0` | positive Gaussian sigma in resized pixels |
| `data.max_train_samples` | `256` | positive or null |
| `data.max_valid_samples` | `64` | positive or null |
| `data.max_test_samples` | null | positive or null; independent from validation |

## `model` and `loss`

| Field | Default | Contract |
|---|---:|---|
| `model.name` | `panoptic_unet_small` | registered model name |
| `model.in_channels` | `3` | image channels |
| `model.expected_num_classes` | `3` | must equal prepared schema count |
| `model.base_channels` | `32` | at least 4 |
| `loss.semantic_weight` | `1.0` | nonnegative CE multiplier |
| `loss.center_weight` | `1.0` | nonnegative center focal multiplier |
| `loss.offset_weight` | `0.01` | nonnegative thing L1 multiplier |
| `loss.ignore_index` | `255` | nonnegative and equal to schema ignore |

## `train`

| Field | Default | Contract |
|---|---:|---|
| `train.epochs` | `20` | positive; may only increase on resume |
| `train.lr` | `0.001` | positive |
| `train.weight_decay` | `0.0001` | nonnegative |
| `train.optimizer` | `adamw` | `adamw` or `sgd` |
| `train.scheduler` | `cosine` | `none`, `cosine`, or `step` |
| `train.scheduler_step_size` | `10` | positive, used by step |
| `train.scheduler_gamma` | `0.1` | `(0,1]`, used by step |
| `train.amp` | `true` | CUDA AMP only |
| `train.grad_clip` | `1.0` | nonnegative; zero disables |
| `train.seed` | `42` | Python, NumPy, torch, loader seed |
| `train.best_metric` | `pq` | `pq`, `sq`, `rq`, `pq_thing`, `pq_stuff` |

## `postprocess`

| Field | Default | Contract |
|---|---:|---|
| `postprocess.center_threshold` | `0.2` | probability in `[0,1]` |
| `postprocess.nms_kernel` | `7` | positive odd integer |
| `postprocess.top_k_centers` | `200` | positive global per-image bound |
| `postprocess.instance_area` | `16` | nonnegative minimum thing pixels |
| `postprocess.stuff_area` | `64` | nonnegative minimum stuff pixels |

Top-level `device` is `auto`, `cpu`, `cuda`, or `mps`; `output_dir` defaults to `artifacts`; `run_name` defaults to `panoptic-unet-small`.

`inspect-data --limit-per-split N` checks metadata and hashes for all splits but decodes only the first `N` rows per split. Use the default unlimited mode before a long or published run.

Use YAML values for CLI overrides:

```bash
--set train.amp=false --set data.max_train_samples=null --set data.image_size='[256,512]'
```
