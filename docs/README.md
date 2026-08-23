# Documentation

[简体中文](README.zh-CN.md) | [Project README](../README.md)

Use the documentation by intent instead of reading it linearly.

## Learn

- [Guided learning path](tutorial/learning-path.md)
- [Panoptic tensors and IDs](tutorial/00-basics.md)
- [Environment and CLI](tutorial/01-environment.md)
- [Data and targets](tutorial/02-data-and-targets.md)
- [Panoptic U-Net](tutorial/03-panoptic-unet.md)
- [Training and resume](tutorial/04-training.md)
- [Evaluation and inference](tutorial/05-evaluation-and-inference.md)

## Understand the system

- [How the pipeline works](concepts/how-it-works.md)
- [Code tour](concepts/code-tour.md)
- [Configuration flow](concepts/configuration-flow.md)
- [Architecture decision](architecture/0001-readable-panoptic-baseline.md)

## Perform a task

- [Complete a Kaggle GPU run](guides/kaggle.md)
- [Convert Cityscapes](guides/cityscapes.md)
- [Use your own data](guides/using-your-data.md)
- [Run controlled experiments](guides/experiments.md)
- [Troubleshoot failures](guides/troubleshooting.md)

## Look up a contract

- [Configuration fields](reference/config-reference.md)
- [Data format](reference/data-format.md)
- [Cityscapes mapping](reference/cityscapes.md)
- [Metrics](reference/metrics.md)
- [Checkpoint schema](reference/checkpoint-schema.md)
- [Recorded synthetic run](recorded-run/README.md)
- [Pending licensed Cityscapes run](recorded-run/cityscapes/README.md)

The tutorials explain concepts, guides give procedures, references define exact behavior, and the recorded run contains evidence from one fixed execution. Keep those responsibilities separate when contributing.
