# 配置参考

[English](config-reference.md) | [配置流](../concepts/configuration-flow.zh-CN.md)

全部字段采用严格模式。下列为 package 默认值，实验 YAML 可覆盖。

## `data`

| 字段 | 默认 | 契约 |
|---|---:|---|
| `data_dir` | `data/raw` | 准备/编排使用的源数据位置 |
| `manifest_dir` | `data/manifests` | prepared CSV、schema、metadata |
| `image_size` | `[256,256]` | `[高,宽]`，正数且被 16 整除 |
| `batch_size` | `4` | 正整数 |
| `num_workers` | `0` | 非负；恢复时可改变的运行参数 |
| `horizontal_flip` | `0.5` | `[0,1]` 概率 |
| `center_sigma` | `8.0` | resize 后像素单位的正 Gaussian sigma |
| `max_train_samples` | `256` | 正整数或 null |
| `max_valid_samples` | `64` | 正整数或 null |
| `max_test_samples` | null | 独立 test 上限 |

## `model` 与 `loss`

| 字段 | 默认 | 契约 |
|---|---:|---|
| `model.name` | `panoptic_unet_small` | 注册模型名 |
| `in_channels` | `3` | 图像通道 |
| `expected_num_classes` | `3` | 等于 schema 类别数 |
| `base_channels` | `32` | 至少 4 |
| `semantic_weight` | `1.0` | 非负 CE 权重 |
| `center_weight` | `1.0` | 非负 center focal 权重 |
| `offset_weight` | `0.01` | 非负 thing L1 权重 |
| `ignore_index` | `255` | 非负且等于 schema ignore |

## `train`

| 字段 | 默认 | 契约 |
|---|---:|---|
| `epochs` | `20` | 正整数，恢复时只能增加 |
| `lr` | `0.001` | 正数 |
| `weight_decay` | `0.0001` | 非负 |
| `optimizer` | `adamw` | `adamw` 或 `sgd` |
| `scheduler` | `cosine` | `none`、`cosine`、`step` |
| `scheduler_step_size` | `10` | step 使用的正整数 |
| `scheduler_gamma` | `0.1` | step 使用的 `(0,1]` |
| `amp` | `true` | 仅 CUDA AMP |
| `grad_clip` | `1.0` | 非负，0 关闭 |
| `seed` | `42` | Python、NumPy、torch、loader seed |
| `best_metric` | `pq` | `pq`、`sq`、`rq`、`pq_thing`、`pq_stuff` |

## `postprocess`

| 字段 | 默认 | 契约 |
|---|---:|---|
| `center_threshold` | `0.2` | `[0,1]` 概率 |
| `nms_kernel` | `7` | 正奇数 |
| `top_k_centers` | `200` | 每图全局正上限 |
| `instance_area` | `16` | thing 最小像素数 |
| `stuff_area` | `64` | stuff 最小像素数 |

顶层 `device` 可为 `auto/cpu/cuda/mps`，`output_dir` 默认为 `artifacts`，`run_name` 默认为 `panoptic-unet-small`。CLI override 使用 YAML 值：

```bash
--set train.amp=false --set data.max_train_samples=null --set data.image_size='[256,512]'
```
