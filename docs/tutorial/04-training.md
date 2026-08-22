# 04: Training, Artifacts, and Resume

[简体中文](04-training.zh-CN.md) | [Previous](03-panoptic-unet.md) | [Next](05-evaluation-and-inference.md)

Training minimizes:

```text
semantic_weight * cross_entropy
+ center_weight * center_focal
+ offset_weight * thing_pixel_L1
```

Center focal loss normalizes by true centers and downweights Gaussian shoulders, avoiding domination by the large background. Offset L1 sees only thing pixels. Gradients are checked for finite total loss and clipped when `grad_clip > 0`.

`adamw` and `sgd` are implemented optimizer choices. Schedulers are `none`, `step`, and `cosine`. CUDA AMP uses a GradScaler; CPU and MPS stay in full precision. The learning rate and every loss component are recorded per epoch.

Validation runs after each epoch. `train.best_metric` selects `best.pt`; supported choices are PQ, SQ, RQ, thing PQ, and stuff PQ. Test data is never used inside training.

Checkpoint schema version 1 stores model, optimizer, scheduler, scaler, epoch, best value, complete metric history, RNG, config, schema, environment, and dataset identity. Saves use a unique temporary file followed by `os.replace`. Loads use `weights_only=True`.

Resume permits a larger epoch target and operational data-path/worker/device changes, but rejects different model, loss, post-processing, schema, dataset identity, optimizer, scheduler, seed, augmentation, and other training semantics. Resume uses the run directory's `last.pt`; `metrics.csv` is appended rather than rewritten.

A dry run performs one real parameter update but writes no normal run artifacts. Use it to catch shape, memory, and finite-loss failures, not to estimate quality.
