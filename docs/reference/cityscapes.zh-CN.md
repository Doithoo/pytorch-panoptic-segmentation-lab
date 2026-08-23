# Cityscapes 参考

[English](cityscapes.md) | [Cityscapes 流程](../guides/cityscapes.zh-CN.md)

## 官方 raw ID 与 train ID

| Raw ID | Train ID | 名称 | 实例 |
|---:|---:|---|:---:|
| 7 | 0 | road | 否 |
| 8 | 1 | sidewalk | 否 |
| 11 | 2 | building | 否 |
| 12 | 3 | wall | 否 |
| 13 | 4 | fence | 否 |
| 17 | 5 | pole | 否 |
| 19 | 6 | traffic light | 否 |
| 20 | 7 | traffic sign | 否 |
| 21 | 8 | vegetation | 否 |
| 22 | 9 | terrain | 否 |
| 23 | 10 | sky | 否 |
| 24 | 11 | person | 是 |
| 25 | 12 | rider | 是 |
| 26 | 13 | car | 是 |
| 27 | 14 | truck | 是 |
| 28 | 15 | bus | 是 |
| 31 | 16 | train | 是 |
| 32 | 17 | motorcycle | 是 |
| 33 | 18 | bicycle | 是 |

其余官方 raw label 在 19 类 train-ID 契约中均映射为 `ignore_index=255`。完整 raw 表和颜色编码于 `panoptic_segmenter.data.cityscapes.CITYSCAPES_CLASSES`。

## 实例编码

Cityscapes `instanceIds` 对单个对象使用 `raw_category_id * 1000 + instance_id`。转换器验证 raw category 与 `labelIds` 一致，并把合法 thing 实例按图从 1 开始重新编号。直接等于 category ID 表示 group/crowd，部分衍生数据使用 instance 后缀 0；schema v1 无法表达 crowd，因此这些像素转换为 ignore。stuff 永远使用 instance 0，使通用 target 不依赖 Cityscapes 的原始数字空间。

## Panoptic 编码

官方格式导出重新使用 raw category ID：

```text
panoptic_id = raw_category_id * 1000 + instance_id
```

stuff 使用 `raw_category_id * 1000`，void 为 0。PNG 使用 RGB 字节存储 ID，JSON 的 `segments_info` 包含 `id`、`category_id`、`area`、`iscrowd=0`。用于 leaderboard 前必须进一步核对 Cityscapes crowd/group 政策。

## Split 策略

- `train` 对应 `leftImg8bit/train` + `gtFine/train`。
- `valid` 对应 `leftImg8bit/val` + `gtFine/val`。
- `test` 因公开 test 标注不可用而保持不可用。

不要随机化这些 split，也不要把 validation 称为 test。
