# 05: Evaluation and Inference

[简体中文](05-evaluation-and-inference.zh-CN.md) | [Previous](04-training.md) | [Metrics reference](../reference/metrics.md)

Evaluation safely rebuilds the saved architecture, loads tensor state, applies saved post-processing, and evaluates a prepared split. `max_test_samples` is independent from validation limits.

The decoder performs center sigmoid, NMS, thresholding, and global top-k selection only on predicted thing pixels. It assigns pixels in chunks to same-class centers. Thing regions below `instance_area`, stuff regions below `stuff_area`, and things with no center become void. This bounds the former `H x W x all_centers` memory risk.

PQ matching uses IoU greater than 0.5. Statistics accumulate per class over the entire split before macro averaging. Predictions with more than half their area on target void are not false positives. Outputs include overall PQ/SQ/RQ, thing/stuff PQ, and per-class values.

The project contract does not encode crowd regions. Dataset-specific official comparisons must add the dataset's crowd/void conversion and validate against its official evaluator. Do not compare this teaching result directly with an official leaderboard number without that adapter.

Prediction pads to the next multiple of 16 and crops outputs back to the exact source dimensions. It writes raw semantic IDs, 16-bit instance IDs, stable schema colors, and a per-instance overlay. Thresholds come from the checkpoint config.

Aggregate metrics answer whether a run improved; overlays answer how it failed. Inspect semantic confusion, missing centers, duplicate centers, merged/split objects, tiny-region filtering, and offset direction separately.
