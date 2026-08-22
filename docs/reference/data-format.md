# Data Format

Each sample consists of three files with the same stem:

- `images/<id>.png`: RGB image.
- `semantic/<id>.png`: indexed class IDs. `255` is ignored.
- `instance/<id>.png`: `0` for stuff and ignored pixels; positive IDs identify thing instances.

A CSV manifest is generated with four columns: `sample_id`, `image_path`, `semantic_path`, and `instance_path`. Paths may be absolute or relative to the manifest directory.

The schema lists contiguous class IDs and marks each class with `isthing`. Thing classes are split into separate regions during PQ matching. Stuff classes are evaluated as one region per class.
