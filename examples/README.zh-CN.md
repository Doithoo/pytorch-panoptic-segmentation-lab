# 示例

[English](README.md) | [学习路线](../docs/tutorial/learning-path.zh-CN.md)

在仓库根目录运行下面的文件。它们只打印少量 Tensor 或 shape，方便和教程中的说明对照。

```bash
uv run python examples/01_panoptic_target.py
uv run python examples/02_model_contract.py
uv run python examples/03_minimal_workflow.py
```

- `01_panoptic_target.py`：为一个对象生成 center heatmap 和 offset target。
- `02_model_contract.py`：创建模型并打印 semantic、center、offset 三个输出的 shape。
- `03_minimal_workflow.py`：生成合成数据、准备数据清单并执行 dry-run。

示例程序每次只解释一个概念。完整的用户流程在 CLI 中，错误输入和边界情况由测试覆盖。
