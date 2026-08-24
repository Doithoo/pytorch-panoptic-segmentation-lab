# Use Your Own Data

[简体中文](using-your-data.zh-CN.md) | [Data format](../reference/data-format.md) | [Add a dataset converter](adding-datasets.md)

Before writing model code, convert a small sample and make sure the labels are correct. A model cannot recover from inconsistent class IDs or instance masks.

## 1. Choose the class IDs

Use contiguous training IDs from `0` to `C-1`. Choose an ignore value outside that range, normally `255`. Mark each class as either thing or stuff:

- things are countable objects such as people, cars, or balls;
- stuff describes regions such as road, sky, field, or background.

## 2. Write the three folders

```text
my-data/
  images/sample-001.jpg
  semantic/sample-001.png
  instance/sample-001.png
```

All three files for a sample use the same stem. The semantic mask stores class IDs. The instance mask stores a positive, image-local ID for thing pixels and zero for stuff or ignored pixels.

Use 16-bit PNG or Pillow integer mode if an instance ID may exceed 255. Resize masks with nearest-neighbor interpolation only.

## 3. Define the schema

```yaml
ignore_index: 255
classes:
  - id: 0
    name: background
    isthing: false
    color: [32, 32, 32]
  - id: 1
    name: person
    isthing: true
    color: [230, 80, 80]
```

Class IDs must be contiguous. Names must be unique. Colors are used only for display.

## 4. Create and inspect the manifests

```bash
uv run panoptic-segment prepare-data \
  --data-dir /path/to/my-data \
  --manifest-dir data/my-manifests \
  --schema configs/my_schema.yaml
uv run panoptic-segment inspect-data --manifest-dir data/my-manifests
uv run python scripts/preview_panoptic.py data/my-manifests/train.csv \
  --output artifacts/my-data-preview.png --limit 8
```

Open the preview. Check that colors align with the image, separate objects have separate IDs, and ignored regions are where you expect them.

If samples come from the same video, scene, patient, or large source image, create `groups.csv` and pass `--group-file`. Do not let closely related samples appear in different splits.

If the dataset already defines train, validation, and test membership, preserve it with a dedicated converter instead of using the random splitter.

## 5. Point a config at the data

Set at least:

```yaml
data:
  manifest_dir: data/my-manifests
model:
  expected_num_classes: 2
loss:
  ignore_index: 255
```

Run one batch before starting a long job:

```bash
uv run panoptic-segment train --config configs/my_experiment.yaml --dry-run
```

## What the automatic checks cannot catch

`inspect-data` can find invalid IDs, mismatched sizes, missing files, and many instance errors. It cannot tell whether class 1 was consistently mislabeled as class 2, whether a polygon is shifted, or whether a split is scientifically appropriate. Always inspect examples and document the source version, class mapping, split policy, and license.
