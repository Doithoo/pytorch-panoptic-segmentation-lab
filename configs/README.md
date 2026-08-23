# Configurations

[简体中文](README.zh-CN.md) | [Configuration reference](../docs/reference/config-reference.md)

| File | Purpose |
|---|---|
| `learning_minimal.yaml` | Two-epoch bounded CPU-friendly workflow check |
| `reference_kaggle.yaml` | Twenty-epoch CUDA synthetic reference job |
| `cityscapes.yaml` | Official train/val Cityscapes workflow |
| `synthetic_schema.yaml` | Three-class teaching schema used by generated data |
| `kaggle_soccer_schema.yaml` | Seven-class schema for the public Kaggle Soccer teaching dataset |
| `kaggle_soccer.yaml` | Ten-epoch GPU-friendly Soccer workflow |

A configuration is strict and complete after defaults are merged. Use `show-config` to inspect the resolved value. The synthetic schema is not a Cityscapes schema and must not be used to label a Cityscapes result.
