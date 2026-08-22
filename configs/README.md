# Configurations

[简体中文](README.zh-CN.md) | [Configuration reference](../docs/reference/config-reference.md)

| File | Purpose |
|---|---|
| `learning_minimal.yaml` | Two-epoch bounded CPU-friendly workflow check |
| `reference_kaggle.yaml` | Twenty-epoch CUDA synthetic reference job |
| `synthetic_schema.yaml` | Three-class teaching schema used by generated data |

A configuration is strict and complete after defaults are merged. Use `show-config` to inspect the resolved value. The synthetic schema is not a Cityscapes schema and must not be used to label a Cityscapes result.
