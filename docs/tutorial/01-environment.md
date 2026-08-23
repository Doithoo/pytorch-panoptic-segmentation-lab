# 01: Environment and CLI

[简体中文](01-environment.zh-CN.md) | [Previous](00-basics.md) | [Next](02-data-and-targets.md)

The supported Python range is 3.10–3.12. `uv` keeps the lockfile and environment consistent:

```bash
uv sync --locked --extra dev
uv run panoptic-segment --version
make check
```

`make check` runs lint, format verification, mypy, tests, coverage reporting, and package build. GPU is optional for learning-minimal; use `--device cpu`, `mps`, or `cuda` to make selection explicit.

The CLI commands are:

| Command | Responsibility |
|---|---|
| `show-config` | Merge defaults, YAML, and `--set`, then print the result |
| `prepare-data` | Pair source files and write deterministic manifests |
| `inspect-data` | Validate prepared labels before allocating training time |
| `train` | Dry-run, start, or resume training |
| `evaluate` | Reload a safe checkpoint and score a prepared split |
| `predict` | Preserve source size and export masks plus visualizations |

Installed wheels do not depend on repository-relative default YAML. `show-config` therefore works from any directory. Repository examples still pass explicit configs so the experiment is visible.

Use `--set key=value` for a small controlled change. YAML parsing means `null`, `true`, numbers, and lists keep their types. Unknown keys and invalid values fail before data loading.
