# 配置文件

[English](README.md) | [配置参考](../docs/reference/config-reference.zh-CN.md)

| 文件 | 用途 |
|---|---|
| `learning_minimal.yaml` | 2 epoch、有样本上限、适合 CPU 的流程检查 |
| `reference_kaggle.yaml` | 20 epoch CUDA 合成数据参考任务 |
| `cityscapes.yaml` | 官方 train/val Cityscapes 流程 |
| `synthetic_schema.yaml` | 生成数据使用的三分类教学 schema |

配置与默认值合并后才是实际运行值，请用 `show-config` 检查。合成 schema 不是 Cityscapes schema，不能用它发布 Cityscapes 结果。
