# Panoptic Segmentation Tutorial

[简体中文](README.zh-CN.md) | [Documentation](../README.md) | [Project README](../../README.md)

Use the README for the shortest runnable workflow. Use this index to choose a chapter by the question you are trying to answer.

| Chapter | Question | Read it when |
|---|---|---|
| [00 Panoptic tensors and IDs](00-basics.md) | What does semantic, instance, and panoptic mean? | Tensor shapes or IDs are unfamiliar |
| [01 Environment and CLI](01-environment.md) | How do I install and select a device? | Commands do not run |
| [02 Data and targets](02-data-and-targets.md) | How do masks become center and offset targets? | Preparing or replacing data |
| [03 Panoptic U-Net](03-panoptic-unet.md) | What do the three model heads predict? | Reading the model code |
| [04 Training and resume](04-training.md) | How are losses, validation, checkpoints, and resume connected? | Before a real experiment |
| [05 Evaluation and inference](05-evaluation-and-inference.md) | How should I interpret PQ and prediction failures? | After training |

A practical order is `00 -> 02 -> 03 -> 04 -> 05`. Keep chapter 01 nearby as a setup reference. Each chapter should end with a result you can run, inspect, or explain from the source.

For task-oriented procedures, use the [guides](../README.md#perform-a-task). For exact contracts, use the [reference pages](../README.md#look-up-a-contract).
