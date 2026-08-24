# 使用自己的数据

[English](using-your-data.md) | [数据格式](../reference/data-format.zh-CN.md) | [添加数据集转换器](adding-datasets.zh-CN.md)

先转换少量样本并确认标签，再开始改模型代码。类别 ID 或实例 mask 一旦不一致，模型本身无法补救。

## 1. 选择类别 ID

训练类别使用从 `0` 到 `C-1` 的连续 ID。选择一个范围之外的忽略值，通常是 `255`。每个类别标记为 thing 或 stuff：

- thing 是人、汽车、足球等可以逐个计数的对象；
- stuff 是道路、天空、球场、背景等区域。

## 2. 写出三个目录

```text
my-data/
  images/sample-001.jpg
  semantic/sample-001.png
  instance/sample-001.png
```

一个样本的三个文件必须使用同一个 stem。semantic mask 保存类别 ID；instance mask 在 thing 像素处保存图像内正整数 ID，在 stuff 和忽略像素处保存 0。

如果 instance ID 可能超过 255，请使用 16-bit PNG 或 Pillow 整数模式。mask 只能使用 nearest-neighbor 缩放。

## 3. 定义 schema

```yaml
ignore_index: 255
classes:
  - id: 0
    name: background
    isthing: false
    color: [32, 32, 32]
  - id: 1
    name: person
    isthing: true
    color: [230, 80, 80]
```

类别 ID 必须连续，名称必须唯一。颜色只用于显示。

## 4. 创建并检查数据清单

```bash
uv run panoptic-segment prepare-data \
  --data-dir /path/to/my-data \
  --manifest-dir data/my-manifests \
  --schema configs/my_schema.yaml
uv run panoptic-segment inspect-data --manifest-dir data/my-manifests
uv run python scripts/preview_panoptic.py data/my-manifests/train.csv \
  --output artifacts/my-data-preview.png --limit 8
```

打开预览图，检查颜色是否对应原图、不同对象是否使用不同 ID、忽略区域是否符合预期。

如果样本来自同一视频、场景、病人或大图裁剪，请创建 `groups.csv` 并传入 `--group-file`，不要让高度相关的样本进入不同 split。

数据集已经有 train、validation、test 划分时，应编写独立转换器保留原划分，不要使用随机切分器。

## 5. 在配置中指定数据

至少设置：

```yaml
data:
  manifest_dir: data/my-manifests
model:
  expected_num_classes: 2
loss:
  ignore_index: 255
```

长时间运行前先检查一个 batch：

```bash
uv run panoptic-segment train --config configs/my_experiment.yaml --dry-run
```

## 自动检查无法发现的问题

`inspect-data` 可以发现非法 ID、尺寸不一致、文件缺失和很多实例标注错误，但它无法判断“类别 1 是否被系统性标成类别 2”、多边形是否整体偏移，也无法判断数据划分是否适合你的研究问题。因此仍要人工查看样本，并记录数据版本、类别映射、划分规则和许可证。
