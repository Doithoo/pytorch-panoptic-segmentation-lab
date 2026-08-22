# Data Format and Prepared Metadata

[简体中文](data-format.zh-CN.md) | [Use your own data](../guides/using-your-data.md)

## Source folders

Each sample consists of same-stem files:

- `images/<id>.png|jpg|jpeg`: RGB-decodable image.
- `semantic/<id>.png`: 2D indexed class IDs; schema ignore is excluded.
- `instance/<id>.png`: 2D integer IDs; `0` for stuff/ignore, positive for things.

Class IDs are contiguous from zero. Class names are unique. Every class has `isthing` and an RGB display color. A positive instance is image-local and maps to exactly one thing class. Thing pixels cannot use zero; stuff/ignore cannot use positive IDs. Crowd is not represented in format version 1.

Use 16-bit PNG or Pillow integer mode if an instance ID can exceed 255. Masks are labels, not color photographs; never apply lossy compression or bilinear interpolation.

## Manifest CSV

Prepared `train.csv`, `valid.csv`, and `test.csv` contain:

```text
sample_id,image_path,semantic_path,instance_path
```

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

Format version 1 records `data_dir`, `seed`, `ratios`, `split_counts`, each manifest SHA-256, schema SHA-256, and `identity`. Identity hashes the prepared manifests and schema. Source image content is decoded by preflight but not byte-hashed; document external dataset versions separately for a published benchmark.

`summary.txt` is a human-readable identity and split-count view. It is not a substitute for `dataset.yaml`.
