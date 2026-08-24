# Data Format and Run Metadata

[简体中文](data-format.zh-CN.md) | [Use your own data](../guides/using-your-data.md)

## Source folders

Each sample consists of same-stem files:

- `images/<id>.png|jpg|jpeg`: RGB-decodable image.
- `semantic/<id>.png`: 2D indexed class IDs; schema ignore is excluded.
- `instance/<id>.png`: 2D integer IDs; `0` for stuff/ignore, positive for things.

Class IDs are contiguous from zero. Class names are unique. Every class has `isthing` and an RGB display color. A positive instance is image-local and maps to exactly one thing class. Thing pixels cannot use zero; stuff/ignore cannot use positive IDs. Crowd is not represented in format version 1.

Images are converted to RGB and normalized with ImageNet mean/std inside `PanopticTransform`. Source images can have any dimensions; training resizes them to the configured `data.image_size`, while masks use nearest-neighbor interpolation.

## Manifest CSV

Prepared `train.csv`, `valid.csv`, and `test.csv` contain:

```text
sample_id,image_path,semantic_path,instance_path
```

When samples come from videos, scenes, or neighboring crops, provide an optional `groups.csv` with `sample_id,group_id` and pass it to `prepare-data --group-file`. Group-aware splitting keeps all members of a group in one split and records the split groups in `dataset.yaml`.

Paths are relative to the manifest directory by default, but the loader accepts absolute paths. IDs may not repeat across splits.

## `schema.yaml`

```yaml
classes:
  - id: 0
    name: road
    isthing: false
    color: [80, 120, 180]
  - id: 1
    name: person
    isthing: true
    color: [230, 80, 80]
ignore_index: 255
```

## `dataset.yaml`

Format version 1 records `data_dir`, `seed`, `ratios`, `split_counts`, each manifest SHA-256, schema SHA-256, and `identity`. Group-aware runs additionally record `grouped_split`, `group_file`, and `split_groups`. Identity hashes the prepared manifests and schema. Source image content is decoded by preflight but not byte-hashed; document external dataset versions separately for a published benchmark.

`summary.txt` is a human-readable identity and split-count view. It is not a substitute for `dataset.yaml`.
