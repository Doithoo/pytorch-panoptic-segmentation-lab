# How Configuration Values Are Chosen

[简体中文](configuration-flow.zh-CN.md) | [Configuration reference](../reference/config-reference.md)

Values are applied in this order:

```text
dataclass defaults < YAML file < repeated --set values < --device
```

For example, the following command starts from package defaults, applies `learning_minimal.yaml`, changes the batch size, and finally forces CPU:

```bash
uv run panoptic-segment show-config \
  --config configs/learning_minimal.yaml \
  --set data.batch_size=4 \
  --device cpu
```

`show-config` prints the final values. Unknown keys and invalid values fail before data loading. Paths become `Path` objects, image size becomes a two-item tuple, and all sections are checked together.

At the start of training, the same final configuration is written to `config.yaml` and stored inside each checkpoint. Evaluation and prediction read it from the checkpoint, including image size and post-processing thresholds. The original YAML shows what you requested; the run's `config.yaml` shows what was actually used.

Some settings also need to agree with the prepared data:

- `model.expected_num_classes` must equal the number of classes in `schema.yaml`;
- `loss.ignore_index` must equal the schema's ignore value;
- resume and evaluation require the current `dataset.yaml` fingerprint to match the checkpoint.

Use a new `run_name` when changing an experiment. Training will not overwrite an existing metrics file. Continue an existing run with `--resume` and its `last.pt` checkpoint.
