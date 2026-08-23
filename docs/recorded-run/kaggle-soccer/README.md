# Kaggle Soccer Run

[简体中文](README.zh-CN.md) | [Public Kaggle guide](../../guides/kaggle-soccer.md) | [Kernel metadata](kernel-metadata.json)

## Status: submitted

This kernel attaches the public `quantigoai/soccer-dataset`, converts its video/COCO polygon annotations into the project contract, creates group-aware manifests, trains the seven-class Panoptic U-Net, and writes aggregate plus per-image evaluation artifacts.

The run is teaching evidence, not an official benchmark. The source dataset is CC-BY-SA-4.0, the converter samples frames with `frame_stride=5`, and train/valid/test are split by source video group.

## Reproduce

```bash
kaggle kernels push -p docs/recorded-run/kaggle-soccer
kaggle kernels status yashowhoo/pytorch-panoptic-segmentation-lab-kaggle-soccer
```

After completion, download:

```bash
kaggle kernels output yashowhoo/pytorch-panoptic-segmentation-lab-kaggle-soccer \
  --file-pattern 'artifacts/.*|kaggle-run-summary.json' -p kaggle-soccer-output
```

The final report should preserve source revision, conversion parameters, group split counts, resolved config, checkpoint hash, aggregate metrics, per-class metrics, and worst cases.
