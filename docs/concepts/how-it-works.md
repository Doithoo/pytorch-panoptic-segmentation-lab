# How the Pipeline Works

[简体中文](how-it-works.zh-CN.md) | [Code tour](code-tour.md)

```text
three source folders
  -> deterministic manifests + schema + identity
  -> full label preflight
  -> synchronized resize/flip
  -> semantic + Gaussian center + offset targets
  -> PanopticUNet three-head outputs
  -> weighted losses and optimizer
  -> bounded class-aware decoding
  -> per-class PQ accumulator
  -> safe checkpoints, metrics, and visual outputs
```

Preparation and training are deliberately separate. A manifest freezes split membership and gives every later stage one row contract. A schema freezes class order, thing/stuff meaning, colors, and ignore ID. The checkpoint embeds both resolved config and schema; evaluation does not infer them from filenames.

Semantic prediction identifies classes. Center/offset prediction only resolves thing identity. This separation means a center cannot repair a wrong semantic class, and perfect semantic logits do not guarantee separated objects.

Post-processing is part of experiment semantics. Center threshold, NMS kernel, top-k, and minimum areas affect PQ and are therefore configured and checkpointed rather than hidden CLI flags.

Validation selects the model; test measures the selected model once. Run metadata and dataset identity make the result auditable. A completed pipeline is still not an official benchmark until its dataset conversion and evaluator match that benchmark's protocol.
