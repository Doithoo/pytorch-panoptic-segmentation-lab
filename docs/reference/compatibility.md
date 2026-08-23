# Compatibility Policy

[简体中文](compatibility.zh-CN.md) | [Checkpoint schema](checkpoint-schema.md) | [Configuration fields](config-reference.md)

## Runtime

The supported Python range is 3.10 through 3.12. The project pins compatible major/minor ranges for PyTorch, torchvision, NumPy, Pillow, and PyYAML in `pyproject.toml`; `uv.lock` is the reproducible development environment. CI checks all three Python versions on CPU.

CUDA and MPS are optional runtime devices. The reference Kaggle workflow is tested on a T4-class CUDA environment. A successful CPU test does not prove that a particular CUDA build contains kernels for a requested GPU architecture.

## Stored artifacts

- Prepared data format: version 1.
- Checkpoint schema: version 1.
- Model names are resolved through the registered factory name in the saved config.
- Evaluation reports are JSON and may use `null` for metrics of absent classes.

A checkpoint can be resumed only when its schema, data identity, model, loss, post-processing, and training semantics match the current run. The current loader rejects unknown checkpoint schema versions rather than guessing a migration.

## Change policy

Changes to labels, target fields, model output heads, post-processing semantics, checkpoint fields, or metric definitions require:

1. a focused compatibility note;
2. fixtures and round-trip tests;
3. a deliberate schema version or migration decision;
4. updates to English and Simplified Chinese contract documents;
5. a recorded limitation when older artifacts cannot be loaded.

Operational settings such as device, worker count, and data paths may change where the checkpoint resume contract permits them. Do not describe a changed benchmark protocol as a continuation of an older run.
