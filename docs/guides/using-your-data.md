# Use Your Own Data

[简体中文](using-your-data.zh-CN.md) | [Data reference](../reference/data-format.md)

1. Choose contiguous training IDs `0..C-1`, one ignore ID outside that range, and stable thing/stuff meaning.
2. Convert images and labels into same-stem `images`, `semantic`, and `instance` folders.
3. Write a schema with unique names, RGB colors, and `isthing` flags.
4. Prepare and inspect before editing model settings.

```bash
uv run panoptic-segment prepare-data --data-dir /path/to/raw \
  --manifest-dir data/my-manifests --schema configs/my_schema.yaml
uv run panoptic-segment inspect-data --manifest-dir data/my-manifests
```

Then set `data.manifest_dir`, `model.expected_num_classes`, and `loss.ignore_index`. Preserve official splits when they exist; the generic preparer creates random independent-sample splits and is not a benchmark converter.

Converters should be separate tested modules that document raw ID mapping, void/crowd behavior, instance encoding, official split sources, and license. Store instance IDs in 16-bit PNG or integer mode when values can exceed 255. Never encode instance boundaries by interpolating colors.

Before a long run, inspect class/instance frequencies and several overlays. A passing structural preflight cannot detect a consistently wrong class map or swapped channel semantics.
