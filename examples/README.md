# Examples

[简体中文](README.zh-CN.md) | [Learning path](../docs/tutorial/learning-path.md)

Run these files from the repository root. They print small tensors or shapes so you can compare the output with the explanation in the tutorial.

```bash
uv run python examples/01_panoptic_target.py
uv run python examples/02_model_contract.py
uv run python examples/03_minimal_workflow.py
```

- `01_panoptic_target.py` builds a center heatmap and offset target for one object.
- `02_model_contract.py` creates a model and prints the shapes of the semantic, center, and offset outputs.
- `03_minimal_workflow.py` creates synthetic data, prepares manifests, and runs a dry run.

Examples show one idea at a time. The CLI contains the complete user path, and the tests cover invalid inputs and edge cases.
