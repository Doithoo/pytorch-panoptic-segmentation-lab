# Metrics Reference

[简体中文](metrics.zh-CN.md) | [Evaluation tutorial](../tutorial/05-evaluation-and-inference.md)

For each class, predicted and target segments match when IoU is strictly greater than `0.5`. One segment can match once.

```text
PQ = sum matched IoU / (TP + 0.5 FP + 0.5 FN)
SQ = sum matched IoU / TP
RQ = TP / (TP + 0.5 FP + 0.5 FN)
PQ = SQ * RQ when denominators are defined
```

Counts and IoU accumulate per class across the full split. Overall `pq`, `sq`, and `rq` are macro means over classes where the corresponding value is defined. `pq_thing` and `pq_stuff` average valid thing and stuff classes separately. `pq:class_<id>`, `sq:class_<id>`, and `rq:class_<id>` expose class values; an absent class can be NaN in Python/YAML and is serialized as null by the Kaggle JSON writer.

A predicted segment unmatched to a target is ignored as an FP when more than half its area overlaps target void. Void pixels are excluded from matching union. The implementation fixes split-level class aggregation and the previous ineffective ignore subtraction.

Limitations: schema v1 has no crowd field or dataset-native segment JSON. These metrics are appropriate for the project's non-crowd three-mask contract. Official Cityscapes/COCO claims require conversion tests against their evaluator and any server policy.

Training `metrics.csv` reports validation metrics after every epoch. `best.pt` uses the configured metric. Final test evaluation should run once after selection and should name the exact checkpoint hash.
