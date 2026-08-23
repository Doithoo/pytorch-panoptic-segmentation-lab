# Cityscapes Reference

[简体中文](cityscapes.zh-CN.md) | [Cityscapes guide](../guides/cityscapes.md)

## Official raw IDs and train IDs

| Raw ID | Train ID | Name | Instance |
|---:|---:|---|:---:|
| 7 | 0 | road | no |
| 8 | 1 | sidewalk | no |
| 11 | 2 | building | no |
| 12 | 3 | wall | no |
| 13 | 4 | fence | no |
| 17 | 5 | pole | no |
| 19 | 6 | traffic light | no |
| 20 | 7 | traffic sign | no |
| 21 | 8 | vegetation | no |
| 22 | 9 | terrain | no |
| 23 | 10 | sky | no |
| 24 | 11 | person | yes |
| 25 | 12 | rider | yes |
| 26 | 13 | car | yes |
| 27 | 14 | truck | yes |
| 28 | 15 | bus | yes |
| 31 | 16 | train | yes |
| 32 | 17 | motorcycle | yes |
| 33 | 18 | bicycle | yes |

All other official raw labels map to `ignore_index=255` in the 19-class train-ID contract. The complete raw label table and official colors are encoded in `panoptic_segmenter.data.cityscapes.CITYSCAPES_CLASSES`.

## Instance encoding

Cityscapes `instanceIds` use `raw_category_id * 1000 + instance_id` for individual objects. The converter verifies that the raw category agrees with `labelIds`, then re-indexes valid thing instances from 1 per image. A bare category ID identifies a group/crowd region (and some derived data uses a zero instance suffix); schema version 1 cannot represent crowd, so those pixels become ignore. Stuff regions always use instance zero. This makes the generic target independent of Cityscapes' raw numeric namespace.

## Panoptic encoding

Official-format exports use the raw category ID again:

```text
panoptic_id = raw_category_id * 1000 + instance_id
```

Stuff uses `raw_category_id * 1000`; void is zero. PNG IDs are stored as RGB bytes, and JSON `segments_info` includes `id`, `category_id`, `area`, and `iscrowd=0`. Cityscapes crowd/group policy must be reviewed before using these exports for a leaderboard claim.

## Split policy

- `train` maps to `leftImg8bit/train` + `gtFine/train`.
- `valid` maps to `leftImg8bit/val` + `gtFine/val`.
- `test` remains unavailable locally because test annotations are not public.

Do not randomize these splits and do not call validation test.
