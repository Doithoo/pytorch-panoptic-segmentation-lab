# 脚本

[English](README.md) | [Kaggle 指南](../docs/guides/kaggle.zh-CN.md)

- `create_synthetic_data.py`：生成结构确定的教学样本，支持 `--count`、`--size` 和 `--output`。
- `kaggle_train.py`：完成 CUDA 预检、数据准备、训练/恢复、test 评估和 Kaggle 摘要。

脚本只编排 package API，训练语义统一放在 `src/panoptic_segmenter`，避免形成第二套无法测试的实现。
