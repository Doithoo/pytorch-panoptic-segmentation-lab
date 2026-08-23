# 02：数据与监督目标

[English](02-data-and-targets.md) | [上一节](01-environment.zh-CN.md) | [下一节](03-panoptic-unet.zh-CN.md)

训练前先生成、准备并检查：

```bash
uv run python scripts/create_synthetic_data.py --count 24 --size 128
uv run panoptic-segment prepare-data --schema configs/synthetic_schema.yaml
uv run panoptic-segment inspect-data
```

下面的 contact sheet 是这一阶段的可视化检查点：训练前应确认原图、semantic 配色和 panoptic overlay 相互一致。

![合成数据全景预览](../assets/synthetic-panoptic-preview.png)

准备过程要求 `images`、`semantic`、`instance` 的 stem 完全一致，并拒绝重复 stem、缺目录、非法比例和无法形成非空 split 的样本数。固定 seed 打乱 ID，每个非零 split 至少得到一个样本，manifest 使用相对路径。

`dataset.yaml` 记录格式版本、相对数据根、seed、比例、split 数、manifest 哈希、schema 哈希和准备身份。该身份用于绑定恢复协议，但不声称逐字节哈希所有源图像。

预检会解码每个样本并检查尺寸、semantic ID、非负 instance、ignore/stuff 为 0、thing 为正 ID、一个实例不跨语义类、manifest 数量和跨 split ID。

图像使用 bilinear resize，mask 使用 nearest；三者先同步变换再构建 target，因此 offset 单位属于缩放后的训练网格。水平翻转也同步作用于三者。

每个有效 thing 实例在像素坐标均值处生成 Gaussian，重叠位置取最大值。offset loss 仅作用于 thing 像素，ignore 不参与 center 和 semantic 监督。

随机 split 适合独立自定义样本和合成教学数据。真实 benchmark 必须保持官方 split 和场景/组边界，不要随机拆分视频相邻帧或同一大图的 crop。
