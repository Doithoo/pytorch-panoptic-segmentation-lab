# Controlled Experiments

[简体中文](experiments.zh-CN.md) | [Configuration flow](../concepts/configuration-flow.md)

Change one explanatory variable at a time and give every run a unique name:

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml \
  --set run_name=center-threshold-015 \
  --set postprocess.center_threshold=0.15
```

Keep data identity, split, seed, model, epoch budget, and selection metric fixed unless one is the variable. Compare resolved `config.yaml`, not only the command you remember. Report best validation epoch, selected metric, final test metrics, environment, sample limits, and at least one visual failure.

Post-processing tuning on validation is model selection. Freeze thresholds before test evaluation. Multiple threshold trials on test leak information and make the final number optimistic.

Use at least three seeds before making a general optimization claim. A two-epoch synthetic comparison demonstrates workflow behavior only. Record mean, variation, runtime, and failures rather than retaining only the winning run.
