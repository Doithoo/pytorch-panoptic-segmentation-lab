# 示例

[English](README.md) | [学习路线](../docs/tutorial/learning-path.zh-CN.md)

在仓库根目录运行：

```bash
uv run python examples/01_panoptic_target.py
uv run python examples/02_model_contract.py
uv run python examples/03_minimal_workflow.py
```

`01` 展示 Gaussian center 和 offset target，`02` 检查模型三头输出，`03` 创建数据、准备 manifest 并执行生产 dry run。示例用于解释，CLI 集成和边界行为由测试负责。
