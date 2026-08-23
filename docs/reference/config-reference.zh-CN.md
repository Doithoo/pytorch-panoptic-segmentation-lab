# 配置参考

[English](config-reference.md) | [配置流](../concepts/configuration-flow.zh-CN.md)

全部字段采用严格模式。下列为 package 默认值，实验 YAML 可覆盖。

## `data`

| 字段 | 默认 | 契约 |
|---|---:|---|
| `data.data_dir` | `data/raw` | 准备/编排使用的源数据位置 |
| `data.manifest_dir` | `data/manifests` | prepared CSV、schema、metadata |
| `data.image_size` | `[256,256]` | `[高,宽]`，正数且被 16 整除；源图像会被 resize 到该网格 |
| `data.batch_size` | `4` | 正整数 |
| `data.num_workers` | `0` | 非负；恢复时可改变的运行参数 |
| `data.horizontal_flip` | `0.5` | `[0,1]` 概率 |
| `data.center_sigma` | `8.0` | resize 后像素单位的正 Gaussian sigma |
| `data.max_train_samples` | `256` | 正整数或 null |
| `data.max_valid_samples` | `64` | 正整数或 null |
| `data.max_test_samples` | null | 独立 test 上限 |

## `model` 与 `loss`

| 字段 | 默认 | 契约 |
|---|---:|---|
| `model.name` | `panoptic_unet_small` | 注册模型名 |
| `model.in_channels` | `3` | 图像通道 |
| `model.expected_num_classes` | `3` | 等于 schema 类别数 |
| `model.base_channels` | `32` | 至少 4 |
| `loss.semantic_weight` | `1.0` | 非负 CE 权重 |
| `loss.center_weight` | `1.0` | 非负 center focal 权重 |
| `loss.offset_weight` | `0.01` | 非负 thing L1 权重 |
| `loss.ignore_index` | `255` | 非负且等于 schema ignore |

## `train`

| 字段 | 默认 | 契约 |
|---|---:|---|
| `train.epochs` | `20` | 正整数，恢复时只能增加 |
| `train.lr` | `0.001` | 正数 |
| `train.weight_decay` | `0.0001` | 非负 |
| `train.optimizer` | `adamw` | `adamw` 或 `sgd` |
| `train.scheduler` | `cosine` | `none`、`cosine`、`step` |
| `train.scheduler_step_size` | `10` | step 使用的正整数 |
| `train.scheduler_gamma` | `0.1` | step 使用的 `(0,1]` |
| `train.amp` | `true` | 仅 CUDA AMP |
| `train.grad_clip` | `1.0` | 非负，0 关闭 |
| `train.seed` | `42` | Python、NumPy、torch、loader seed |
| `train.best_metric` | `pq` | `pq`、`sq`、`rq`、`pq_thing`、`pq_stuff` |

## `postprocess`

| 字段 | 默认 | 契约 |
|---|---:|---|
| `postprocess.center_threshold` | `0.2` | `[0,1]` 概率 |
| `postprocess.nms_kernel` | `7` | 正奇数 |
| `postprocess.top_k_centers` | `200` | 每图全局正上限 |
| `postprocess.instance_area` | `16` | thing 最小像素数 |
| `postprocess.stuff_area` | `64` | stuff 最小像素数 |

`inspect-data --limit-per-split N` 会检查所有 split 的元数据和 hash，但每个 split 只解码前 `N` 行。长时间运行或发布结果前应使用默认的无限制模式。

顶层 `device` 可为 `auto/cpu/cuda/mps`，`output_dir` 默认为 `artifacts`，`run_name` 默认为 `panoptic-unet-small`。CLI override 使用 YAML 类型：

```bash
--set train.amp=false --set data.max_train_samples=null --set data.image_size='[256,512]'
```
