# Configuration Flow

[简体中文](configuration-flow.zh-CN.md) | [Configuration reference](../reference/config-reference.md)

Resolution order is:

```text
dataclass defaults < YAML values < repeated --set values < explicit --device
```

Unknown YAML and override fields fail. Paths become `Path`, image size becomes a tuple, and all dataclass sections are reconstructed before validation. `show-config` prints this exact resolved object.

The resolved config is written to the run and embedded in every checkpoint. Evaluation and prediction recover it from the checkpoint, including post-processing thresholds. The source YAML is an input; `config.yaml` is the authoritative record of what the run used.

Some facts deliberately have a second external check: model class count must match the prepared schema, loss ignore ID must match schema ignore ID, and resume dataset identity must match `dataset.yaml`. These checks prevent a syntactically valid configuration from changing label meaning.

Use separate run names for experiments. A normal train refuses to overwrite an existing metrics file; `--resume` is required to continue it.
