# 05: Evaluation and Inference

[简体中文](05-evaluation-and-inference.zh-CN.md) | [Previous](04-training.md) | [Metrics reference](../reference/metrics.md)

Evaluation first validates the prepared manifests and compares their `dataset.yaml` identity with the checkpoint identity. It then safely rebuilds the saved architecture, loads tensor state, applies saved post-processing, and evaluates a prepared split. `max_test_samples` is independent from validation limits.

The decoder performs center sigmoid, NMS, thresholding, and global top-k selection only on predicted thing pixels. It assigns pixels in chunks to same-class centers. Thing regions below `instance_area`, stuff regions below `stuff_area`, and things with no center become void. This bounds the former `H x W x all_centers` memory risk.

PQ matching uses IoU greater than 0.5. Statistics accumulate per class over the entire split before macro averaging. Predictions with more than half their area on target void are not false positives. Outputs include overall PQ/SQ/RQ, thing/stuff PQ, and per-class values.

The project contract does not encode crowd regions. Dataset-specific official comparisons must add the dataset's crowd/void conversion and validate against its official evaluator. Do not compare this teaching result directly with an official leaderboard number without that adapter.

Prediction resizes the source to the checkpoint's training `data.image_size`, decodes there, and restores discrete masks to the exact source dimensions with nearest-neighbor interpolation. It writes raw semantic IDs, 16-bit instance IDs, stable schema colors, and a per-instance overlay. Thresholds come from the checkpoint config.

The CLI can persist an auditable report next to the run:

```bash
uv run panoptic-segment evaluate artifacts/run/best.pt --split test \
  --output artifacts/run/evaluation.json
```

The report includes the checkpoint SHA-256, requested and resolved device, split, prepared-data identity, aggregate metrics, per-image metrics, and a lowest-PQ worst-case shortlist. Inspect semantic confusion, missing centers, duplicate centers, merged/split objects, tiny-region filtering, and offset direction separately.
