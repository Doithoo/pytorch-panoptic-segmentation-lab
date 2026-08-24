# Troubleshooting

[简体中文](troubleshooting.zh-CN.md)

| Symptom | What to check |
|---|---|
| `dataset stems do not match exactly` | Compare filenames in `images`, `semantic`, and `instance`. The preparer does not silently drop a file. |
| `split is empty` | Add samples or change ratios. For grouped data, make sure there are enough groups. |
| Group appears in two splits | Check `groups.csv`; one video, scene, patient, or source image must have one group ID. |
| Thing/stuff inspection error | Fix the converter. Thing pixels need positive instance IDs; stuff and void pixels need zero. |
| Image size is not divisible by 16 | Change the training `data.image_size` to values such as 128, 256, or `[256,512]`. Source images may have other sizes. |
| Class count or ignore mismatch | Compare the config with `schema.yaml`. |
| Run already exists | Choose a new `run_name` or resume its `last.pt`. |
| Resume or evaluation identity mismatch | Use the original manifests and schema, or start a new run for the changed data. |
| Checkpoint cannot be loaded | It may be corrupt or use another schema version. Do not switch to `weights_only=False` for an untrusted file. |
| CUDA is unavailable | Select an accelerator and restart the process. |
| P100 CUDA kernel error | Use a T4 or newer GPU. |
| Out of memory | Lower batch size, training image size, or `base_channels`. |
| PQ is zero but semantic output looks plausible | Inspect center peaks, threshold, offsets, thing flags, and area thresholds. |
| Predictions are void | No same-class center survived, or the assigned region was filtered by area. |
| Wheel command cannot find data | A wheel can show defaults from any directory, but training still needs prepared manifests at the configured path. |

Before a long run, execute `inspect-data`, open a preview, and run `train --dry-run`. When reporting a failure, include the command, final config, traceback, Python/PyTorch versions, device, and one non-private sample.
