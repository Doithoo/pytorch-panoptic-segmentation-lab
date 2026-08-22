# Panoptic Segmentation Learning Path

[简体中文](learning-path.zh-CN.md) | [Documentation](../README.md)

This 8–12 hour path assumes basic tensors, convolutions, and gradient descent. Complete each command and explain its output before starting the Kaggle run.

## 1. Verify the environment

```bash
uv sync --extra dev
uv run panoptic-segment --version
uv run panoptic-segment show-config --config configs/learning_minimal.yaml
make check
```

Identify image size, sample limits, three loss weights, post-processing limits, device, and best metric in the resolved configuration.

## 2. Inspect the target contract

```bash
uv run python scripts/create_synthetic_data.py --count 24 --size 128
uv run panoptic-segment prepare-data --schema configs/synthetic_schema.yaml
uv run panoptic-segment inspect-data
uv run python examples/01_panoptic_target.py
```

Explain why semantic is `[H,W]`, center is `[H,W]`, offset is `[2,H,W]`, and instance is not fed directly to the model. Check that thing pixels have positive instance IDs while stuff and void use zero.

## 3. Follow one model update

```bash
uv run python examples/02_model_contract.py
uv run panoptic-segment train --config configs/learning_minimal.yaml --dry-run
```

The dry run performs forward, finite-loss validation, backward, gradient clipping, and one optimizer step. It does not publish a normal checkpoint or metric.

## 4. Complete and inspect a run

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml
```

Read `config.yaml`, `run.yaml`, `metrics.csv`, `best.pt`, and `last.pt` in that order. Explain why best and last can be different and why test data must not select the checkpoint.

## 5. Evaluate and return to pixels

```bash
uv run panoptic-segment evaluate artifacts/learning-minimal/best.pt --split test
uv run panoptic-segment predict artifacts/learning-minimal/best.pt \
  data/raw/images/sample_0000.png
```

Compare overall PQ with `pq_thing` and `pq_stuff`, then inspect the color mask and overlay. A low PQ can come from semantic errors, missed centers, duplicate centers, poor offsets, or area filtering.

## 6. Resume one controlled experiment

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml \
  --set train.epochs=4 --resume artifacts/learning-minimal/last.pt
```

Resume restores model, optimizer, scheduler, scaler, RNG, metric history, and best value. Changing the model, loss, post-processing, schema, or data identity is rejected.

## 7. Submit the Kaggle reference job

Follow the [Kaggle guide](../guides/kaggle.md). The first job uses deterministic synthetic data to prove the non-interactive GPU workflow. Treat it as systems evidence, not a real-world benchmark.

You are ready to extend the project when you can explain Gaussian center supervision, thing-only offsets, void handling, per-class PQ accumulation, validation selection, safe checkpoint loading, and the difference between a workflow record and a benchmark claim.
