# 02：数据与监督目标

[English](02-data-and-targets.md) | [上一节](01-environment.zh-CN.md) | [下一节](03-panoptic-unet.zh-CN.md)

先用生成数据，这样每个标签都可以打开查看：

```bash
uv run python scripts/create_synthetic_data.py --count 24 --size 128
uv run panoptic-segment prepare-data --schema configs/synthetic_schema.yaml
uv run panoptic-segment inspect-data
uv run python scripts/preview_panoptic.py data/manifests/train.csv \
  --output artifacts/target-preview.png --limit 4
```

三个源目录必须包含完全一致的 stem。准备阶段会拒绝重复 stem、缺少目录、非法比例，以及无法满足 split 数量要求的数据。固定 seed 控制随机切分。视频帧或相邻裁剪存在关联时，改用 `--group-file`。

`dataset.yaml` 保存源路径、seed、比例、split 数、文件哈希、schema 哈希、数据指纹；使用 group split 时，还会保存 group 文件和各 split 使用的 group。它不会对所有源图像逐字节哈希。

`inspect-data` 会打开样本并检查：

- 图像、semantic 和 instance 的尺寸；
- 已声明的 semantic ID 和 ignore 值；
- 非负 instance ID；
- stuff 与忽略像素是否使用 0；
- thing 像素是否使用正整数 ID；
- 一个正 instance 是否只对应一个语义类别；
- manifest 数量、哈希和 split 成员；
- group 文件存在时的分组关系。

图像使用 bilinear 插值，mask 使用 nearest-neighbor。三者先执行相同的缩放和水平翻转，再生成监督目标，因此 offset 的单位是缩放后训练网格中的像素。

对于每个有效 thing 实例，`build_targets` 会在实例平均像素位置放置 Gaussian 峰值，并为每个 thing 像素保存 `[dy, dx]`。center loss 不会让大量背景肩部像素主导训练；offset loss 只使用 thing 像素；ignore 像素不参与 semantic 和 center 监督。

独立的合成样本可以随机切分。benchmark 数据应保留官方划分；视频、场景、病人或同一大图裁剪出来的样本应按 group 切分。
