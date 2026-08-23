# Cityscapes 流程

[English](cityscapes.md) | [数据参考](../reference/cityscapes.zh-CN.md) | [Kaggle 指南](kaggle.zh-CN.md)

Cityscapes 是有许可限制的数据集。请接受官方条款后从官网下载；仓库不分发数据，也不会自动下载。

## 解压目录

```text
/path/to/cityscapes/
  leftImg8bit/train/<city>/*_leftImg8bit.png
  leftImg8bit/val/<city>/*_leftImg8bit.png
  gtFine/train/<city>/*_gtFine_labelIds.png
  gtFine/train/<city>/*_gtFine_instanceIds.png
  gtFine/val/<city>/*_gtFine_labelIds.png
  gtFine/val/<city>/*_gtFine_instanceIds.png
```

转换并保留官方 split：

```bash
uv run panoptic-segment convert-cityscapes \
  --data-root /path/to/cityscapes \
  --output-root data/cityscapes
uv run panoptic-segment inspect-data --manifest-dir data/cityscapes
uv run panoptic-segment show-config --config configs/cityscapes.yaml
uv run panoptic-segment train --config configs/cityscapes.yaml \
  --set data.manifest_dir=data/cityscapes \
  --set data.data_dir=data/cityscapes
```

转换器把 raw label ID 映射到 19 个连续 train ID，解码 `category_id * 1000 + instance_id`，按图重新编号实例；`train.csv` 来自官方 train，`valid.csv` 来自官方 val。公开 Cityscapes 没有 test 标注，因此 `test.csv` 明确为空，`inspect-data` 会识别这个声明。

如果不想复制大型 RGB 图像且同一文件系统支持链接，可使用 `--symlink-images`。默认严格模式；`--non-strict` 只用于调查坏标签，不能用于发布结果。

## 官方格式产物

转换器还会写入：

```text
panoptic_train.json
panoptic_valid.json
panoptic/train/*.png
panoptic/valid/*.png
```

PNG 使用 `category_id * 1000 + instance_id` 的标准 RGB 编码，JSON 包含 `images`、`annotations`、`segments_info` 和 19 个类别。这是连接官方 evaluator 的桥梁；项目训练评估器仍使用三 mask。

正式比较时，在独立 benchmark 环境安装并固定 `cityscapesscripts`。先直接从原始 `instanceIds` 生成保留 crowd 的 ground truth：

```bash
uv run --with cityscapesscripts python scripts/evaluate_cityscapes.py prepare-ground-truth \
  --cityscapes-root /path/to/cityscapes \
  --output data/cityscapes-official-panoptic --split val
```

再从选中的 checkpoint 导出预测：

```bash
uv run python scripts/predict_cityscapes.py artifacts/cityscapes-panoptic-unet/best.pt \
  --manifest data/cityscapes/valid.csv \
  --output artifacts/cityscapes-predictions --device cuda
```

运行官方 evaluator：

```bash
uv run --with cityscapesscripts python scripts/evaluate_cityscapes.py evaluate \
  --ground-truth-json data/cityscapes-official-panoptic/cityscapes_panoptic_val.json \
  --prediction-json artifacts/cityscapes-predictions/predictions.json \
  --ground-truth-folder data/cityscapes-official-panoptic/cityscapes_panoptic_val \
  --prediction-folder artifacts/cityscapes-predictions/panoptic \
  --results artifacts/cityscapes-official-results.json
```

通用 `scripts/evaluate_panopticapi.py` 仍可用于其他 COCO-panoptic-compatible 协议，但 Cityscapes 结果应使用 `cityscapesscripts`。记录 package 版本和评估政策；内部 PQ 不能因为缩写相同就被称为官方结果。

## 数据与评估边界

- 官方 train/val 已固定；不要对转换目录再次执行 `prepare-data`，否则会随机重排。
- raw ID 0–33 不是模型类别 ID，必须使用转换器 schema。
- ignore 映射为 255，instance 为 0。
- `caravan`、`trailer` 等 train ID 为 255 的 raw label 被忽略。
- Cityscapes test 标注不公开，应报告 validation 或使用官方服务器。
- 在运行记录中保留原始许可和数据版本。
