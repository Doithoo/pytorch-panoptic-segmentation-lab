# CLI 参考

[English](cli.md) | [环境和 CLI 教程](../tutorial/01-environment.zh-CN.md) | [项目首页](../../README.zh-CN.md)

安装后的命令是 `panoptic-segment`。运行 `panoptic-segment <command> --help` 可查看 argparse 的详细参数。`--set` 使用 `KEY=VALUE`，并由 PyYAML 解析，因此 `null`、布尔值、数字和列表会保留类型。

## `show-config`

打印默认值、YAML 文件和命令行覆盖合并后的配置，不访问数据。

```bash
uv run panoptic-segment show-config --config configs/learning_minimal.yaml
uv run panoptic-segment show-config --set data.image_size='[256,512]' --set train.amp=false
```

## `prepare-data`

按完全一致的 stem 配对 `images/`、`semantic/` 和 `instance/`，生成 `train.csv`、`valid.csv`、`test.csv`、`schema.yaml` 和 `dataset.yaml`。

```bash
uv run panoptic-segment prepare-data \
  --data-dir data/raw --manifest-dir data/manifests \
  --schema configs/synthetic_schema.yaml --ratios 0.8 0.1 0.1 --seed 42
```

对于视频、场景或 crop 派生数据，可以传入包含 `sample_id,group_id` 的 `--group-file`，让相关样本保持在同一 split：

```bash
uv run panoptic-segment prepare-data \
  --data-dir data/raw --manifest-dir data/manifests \
  --group-file data/raw/groups.csv \
  --schema configs/synthetic_schema.yaml
```

通用命令默认随机切分。带官方 split 的 benchmark 不应使用它，而应使用对应转换器。

## `inspect-data`

训练前检查 manifest 元数据和全景标签。

```bash
uv run panoptic-segment inspect-data --manifest-dir data/manifests
uv run panoptic-segment inspect-data --manifest-dir data/manifests --limit-per-split 8
```

`--limit-per-split` 只限制解码内容检查数量；元数据、manifest hash、split 数量和跨 split ID 仍会检查，但通过限定检查不代表所有样本都已验证。

## `convert-cityscapes`

转换有许可的 Cityscapes `labelIds` 和 `instanceIds`，并保留官方 train/val 成员。命令会生成通用 mask 和官方格式 panoptic 产物。

```bash
uv run panoptic-segment convert-cityscapes \
  --data-root /path/to/cityscapes --output-root data/cityscapes
```

同一文件系统上可以使用 `--symlink-images` 避免复制 RGB 文件。`--non-strict` 只应在排查错误源标签时使用。

## `train`

启动实验、用 `--dry-run` 执行一次真实 batch 更新，或从配置 run 目录中的 `last.pt` 恢复。

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml --device cpu --dry-run
uv run panoptic-segment train --config configs/learning_minimal.yaml --set run_name=first-run
uv run panoptic-segment train --config configs/learning_minimal.yaml \
  --set train.epochs=4 --resume artifacts/first-run/last.pt
```

普通训练不会覆盖已有 run。恢复允许增加 `train.epochs`，也允许修改路径、worker 和设备等运行参数，但会拒绝改变模型、数据语义、schema、loss、optimizer、scheduler、增强、后处理等设置。

## `evaluate`

重载 checkpoint，校验准备数据的 identity，并评估一个 split，默认是 `valid`。

```bash
uv run panoptic-segment evaluate artifacts/first-run/best.pt --split valid --device cpu
uv run panoptic-segment evaluate artifacts/first-run/best.pt --split test \
  --output artifacts/first-run/evaluation.json
```

可选 JSON 报告会记录 checkpoint 路径和 SHA-256、split、设备、数据 identity、总体指标、每个样本一行的指标以及 PQ 最低的 `worst_cases` 列表。使用 `--worst-cases 0` 可以省略该 shortlist。评估使用 checkpoint 中保存的后处理设置。

## `predict`

对一张 RGB 图像执行保存尺度的推理，并把离散 mask 恢复到源图像尺寸。

```bash
uv run panoptic-segment predict artifacts/first-run/best.pt \
  data/raw/images/sample_0000.png --output artifacts/prediction --device cpu
```

输出目录包含 semantic ID、16-bit instance ID、schema 配色图和 overlay。Cityscapes 官方格式导出应使用 `scripts/predict_cityscapes.py`。

## 相关脚本

- `scripts/preview_panoptic.py`：生成原图/semantic/panoptic contact sheet。
- `scripts/convert_kaggle_soccer.py`：转换公共 Kaggle Soccer 数据集。
- `scripts/convert_cityscapes.py`：有许可的 Cityscapes 转换。
- `scripts/predict_cityscapes.py`：Cityscapes panoptic 预测导出。
- `scripts/evaluate_cityscapes.py`：保留 crowd 语义的官方评估包装器。
- `scripts/evaluate_panopticapi.py`：可选 COCO-panoptic-compatible 评估器。
- `scripts/kaggle_train.py`：合成数据 GPU 参考运行器。

脚本依赖和 benchmark 边界见 [Scripts](../../scripts/README.zh-CN.md) 以及 [Cityscapes 指南](../guides/cityscapes.zh-CN.md)。
