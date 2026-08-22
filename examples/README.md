# Examples

[简体中文](README.zh-CN.md) | [Learning path](../docs/tutorial/learning-path.md)

Run examples from the repository root:

```bash
uv run python examples/01_panoptic_target.py
uv run python examples/02_model_contract.py
uv run python examples/03_minimal_workflow.py
```

`01` exposes Gaussian center and offset targets. `02` checks the three model heads. `03` creates data, prepares manifests, and executes a production dry run. Examples are small explanations; CLI integration and edge cases live in tests.
