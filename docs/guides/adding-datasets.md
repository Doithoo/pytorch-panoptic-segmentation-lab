# Adding A Dataset Converter

[简体中文](adding-datasets.zh-CN.md) | [Data format](../reference/data-format.md) | [Use your own data](using-your-data.md)

Use the generic `prepare-data` command only when samples are independent and a random split is acceptable. A benchmark or video-derived dataset should have a dedicated converter that preserves its official split and protocol.

## Converter contract

A converter should produce the standard folders or equivalent manifest rows:

```text
images/<sample>.png
semantic/<sample>.png
instance/<sample>.png
train.csv
valid.csv
test.csv
schema.yaml
dataset.yaml
```

It must document and test:

- raw-to-contiguous class ID mapping;
- thing/stuff/crowd/void behavior;
- instance encoding and re-indexing;
- official split membership and source version;
- image and mask dimensions, dtypes, and interpolation rules;
- license and redistribution boundaries;
- panoptic JSON/PNG export rules when an official evaluator needs them.

## Steps

1. Keep raw data outside Git and make the source root explicit.
2. Convert one hand-checked sample before processing the full dataset.
3. Add malformed-label fixtures: unknown IDs, mismatched dimensions, invalid instance category, group/crowd regions, and non-positive thing IDs.
4. Preserve provider sample IDs in manifest columns when official evaluators need them.
5. Register the converter with `register_converter()` when it is a reusable provider.
6. Run `inspect-data` and inspect semantic/instance/overlay previews.
7. Record the dataset version, split policy, license, and converter revision with every run.

Do not randomize official splits by calling `prepare-data` on an already converted benchmark directory. Do not call an internal PQ result an official score without evaluating the exported files through the dataset's own policy.
