# Documentation

[简体中文](README.zh-CN.md) | [Project README](../README.md)

Start with the project README if you want to run the code immediately. Use this page to find the explanation or procedure that matches your current task.

## First run

| Goal | Read |
|---|---|
| Complete the shortest CPU workflow | [Learning path](tutorial/learning-path.md) |
| Choose one tutorial chapter | [Tutorial index](tutorial/README.md) |
| Convert and train on public Soccer data | [Kaggle Soccer](guides/kaggle-soccer.md) |
| See measured outputs from completed runs | [Synthetic run](recorded-run/README.md) and [Soccer run](recorded-run/kaggle-soccer/README.md) |

## Tutorials

| Question | Chapter |
|---|---|
| How are semantic and instance IDs represented? | [Panoptic tensors and IDs](tutorial/00-basics.md) |
| How do I install the project and choose a device? | [Environment and CLI](tutorial/01-environment.md) |
| How do masks become center and offset supervision? | [Data and targets](tutorial/02-data-and-targets.md) |
| What does each model output mean? | [Panoptic U-Net](tutorial/03-panoptic-unet.md) |
| What happens during training and resume? | [Training and resume](tutorial/04-training.md) |
| How are PQ and prediction files produced? | [Evaluation and inference](tutorial/05-evaluation-and-inference.md) |

## Common tasks

| Task | Guide |
|---|---|
| Use the public Soccer dataset | [Kaggle Soccer](guides/kaggle-soccer.md) |
| Run a synthetic job on Kaggle | [Kaggle GPU](guides/kaggle.md) |
| Convert Cityscapes | [Cityscapes](guides/cityscapes.md) |
| Prepare another dataset | [Use your own data](guides/using-your-data.md) |
| Write a dataset converter | [Add a dataset](guides/adding-datasets.md) |
| Register another model | [Add a model](guides/adding-models.md) |
| Compare runs | [Experiments](guides/experiments.md) |
| Diagnose an error or poor result | [Troubleshooting](guides/troubleshooting.md) |

## Understand the code

- [How one sample moves through the pipeline](concepts/how-it-works.md)
- [Source code tour](concepts/code-tour.md)
- [How configuration values are merged](concepts/configuration-flow.md)
- [Why the baseline uses semantic, center, and offset outputs](architecture/0001-readable-panoptic-baseline.md)

## Reference

Use these pages when you need an exact field name, file layout, or formula:

- [CLI commands](reference/cli.md)
- [Configuration fields](reference/config-reference.md)
- [Data and manifest format](reference/data-format.md)
- [PQ, SQ, and RQ](reference/metrics.md)
- [Checkpoint fields and resume rules](reference/checkpoint-schema.md)
- [Cityscapes class mapping](reference/cityscapes.md)
- [Runtime and artifact compatibility](reference/compatibility.md)

Tutorials explain ideas, guides give procedures, reference pages define exact behavior, and recorded runs contain measured outputs from fixed commands.
