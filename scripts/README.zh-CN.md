# Scripts

[English](README.md) | [CLI 参考](../docs/reference/cli.zh-CN.md) | [Cityscapes 指南](../docs/guides/cityscapes.zh-CN.md)

脚本只负责编排 package API。训练语义属于 `src/panoptic_segmenter`，因此脚本应保持轻量、可测试，不形成第二套实现。

| 脚本 | 用途 | 额外要求 |
|---|---|---|
| `create_synthetic_data.py` | 生成确定性教学样本 | package 依赖 |
| `convert_kaggle_soccer.py` | 转换公共 Kaggle Soccer 视频/COCO 标注 | `opencv-python-headless` |
| `preview_panoptic.py` | 生成原图/semantic/panoptic contact sheet | package 依赖 |
| `convert_cityscapes.py` | 转换有许可的 Cityscapes train/val 数据 | Cityscapes 压缩包 |
| `predict_cityscapes.py` | 导出 Cityscapes panoptic PNG/JSON | 已转换的 Cityscapes 数据 |
| `evaluate_cityscapes.py` | 生成 crowd-aware GT 或执行官方验证 | 通过 `uv run --with` 使用 `cityscapesscripts` |
| `evaluate_panopticapi.py` | 运行通用 panoptic evaluator | 独立环境中的 `panopticapi` |
| `kaggle_train.py` | 执行合成数据 CUDA 参考流程 | Kaggle GPU 和 Internet |
| `kaggle_cityscapes.py` | 执行有许可私有数据 Cityscapes 流程 | 私有 Kaggle 数据集、GPU、许可证 |

## 本地示例

```bash
uv run python scripts/create_synthetic_data.py --count 24 --size 128
uv run --with opencv-python-headless python scripts/convert_kaggle_soccer.py --help
uv run python scripts/preview_panoptic.py data/manifests/train.csv \
  --output artifacts/dataset-preview.png --limit 4
uv run python scripts/convert_cityscapes.py --help
uv run python scripts/predict_cityscapes.py --help
uv run python scripts/evaluate_cityscapes.py --help
```

通用三目录流程应使用已安装 CLI。Cityscapes 脚本保留官方 split 和 provider ID；不要用随机 `prepare-data` 替代它们。

## 官方评估

只为当前 benchmark 命令安装可选 evaluator，并在记录结果时保留版本和策略。内部 non-crowd PQ 不能称为官方 Cityscapes 或 COCO 分数。

```bash
uv run --with cityscapesscripts python scripts/evaluate_cityscapes.py --help
uv run python scripts/evaluate_panopticapi.py --help
```

不要提交数据集、凭据、Kaggle token、checkpoint 或生成的 run 目录。信任边界见 [SECURITY.md](../SECURITY.md)。
