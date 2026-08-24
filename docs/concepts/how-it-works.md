# How One Sample Moves Through the Project

[简体中文](how-it-works.zh-CN.md) | [Code tour](code-tour.md)

```text
image + semantic mask + instance mask
  -> fixed train/valid/test manifests
  -> label and file checks
  -> synchronized resize and horizontal flip
  -> semantic, center-heatmap, and offset supervision
  -> three model outputs
  -> weighted losses and parameter updates
  -> center detection and instance assignment
  -> PQ/SQ/RQ calculation
  -> checkpoints, metrics, and prediction images
```

## Before training

`prepare-data` matches files by stem and writes CSV manifests. Once those files exist, every later command uses the same sample membership. If frames or crops are related, a group file keeps them in one split.

`schema.yaml` defines class order, display colors, the ignore value, and whether each class is a thing or stuff class. `inspect-data` then opens the files and checks dimensions, label values, instance IDs, hashes, and split membership.

## During training

The image and both masks receive the same resize and flip. Targets are created after this transform, so center coordinates and offsets use the resized pixel grid.

The model has three outputs:

- semantic logits answer which class each pixel belongs to;
- the center heatmap locates thing instances;
- offsets point each thing pixel toward its instance center.

Semantic prediction and instance separation solve different parts of the problem. Correct semantic logits do not guarantee that two nearby people will be separated, and a good center prediction cannot repair the wrong semantic class.

## After the forward pass

Training combines semantic cross-entropy, center focal loss, and thing-only offset L1 loss. Validation runs after each epoch and selects `best.pt` using `train.best_metric`.

For evaluation and prediction, the decoder finds center peaks, assigns thing pixels to same-class centers, removes regions below the configured area thresholds, and leaves stuff pixels with instance ID 0. Center threshold, NMS size, top-k, and area thresholds are saved in the checkpoint because changing them can change PQ.

The test split is evaluated after model and threshold choices have been made on validation data. `run.yaml`, `config.yaml`, `dataset.yaml`, and the checkpoint hash make it possible to tell exactly which code, data split, and settings produced a result.
