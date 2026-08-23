# 脚本

[English](README.md) | [Kaggle 指南](../docs/guides/kaggle.zh-CN.md)

- `create_synthetic_data.py`：生成结构确定的教学样本，支持 `--count`、`--size` 和 `--output`。
- `kaggle_train.py`：CUDA 合成数据参考流程。
- `kaggle_cityscapes.py`：私有许可数据的 Cityscapes 转换、训练、预测和官方 validation。
- `convert_cityscapes.py`：按官方 split 转换有许可的 Cityscapes，并生成 panoptic JSON/PNG。
- `evaluate_panopticapi.py`：可选通用 panoptic evaluator wrapper。
- `evaluate_cityscapes.py`：生成保留 crowd 的官方 GT 并运行 `cityscapesscripts`。
- `predict_cityscapes.py`：把 checkpoint 的 validation 预测导出为 panoptic PNG/JSON。

脚本只编排 package API，训练语义统一放在 `src/panoptic_segmenter`，避免形成第二套无法测试的实现。
