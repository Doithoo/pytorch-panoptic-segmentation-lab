# Troubleshooting

[简体中文](troubleshooting.zh-CN.md)

| Symptom | Cause and action |
|---|---|
| `dataset stems do not match exactly` | Compare filenames in all three source folders; preparation never silently drops unmatched samples. |
| `split is empty` | Add samples or change ratios; every requested split must be nonempty. |
| thing/stuff preflight issue | Fix converter semantics; do not suppress the check. |
| image size not divisible by 16 | use dimensions such as 128, 256, or `[256,512]`. |
| class count/ignore mismatch | align config with prepared `schema.yaml`. |
| run already exists | choose a new `run_name` or use a compatible `--resume`. |
| resume identity/config mismatch | resume the original protocol; start a new run for a changed experiment. |
| safe checkpoint load failure | file is corrupt, from an older schema, or requires unsafe pickle globals; do not switch to `weights_only=False` for untrusted input. |
| CUDA unavailable | enable an accelerator and restart the process. |
| CUDA kernel error on P100 | choose T4 or newer. |
| CUDA OOM | lower batch size, image size, or base channels; avoid raising center top-k without profiling. |
| PQ is zero but semantic looks plausible | inspect center heatmaps, threshold, offsets, thing flags, and minimum areas. |
| predictions are void | no valid same-class center survived or region area was filtered. |
| wheel CLI cannot find data | wheel defaults no longer require YAML, but training still requires prepared manifests in the configured path. |

Run `inspect-data`, then a production `--dry-run`, before debugging a full job. Include resolved config, complete traceback, environment versions, and one nonprivate sample when reporting a bug.
