# Adding a Model

[简体中文](adding-models.zh-CN.md) | [Model outputs](../tutorial/03-panoptic-unet.md) | [Code tour](../concepts/code-tour.md)

The training loop expects a model that returns three tensors:

```python
{
    "semantic": Tensor[B, C, H, W],
    "center": Tensor[B, 1, H, W],
    "offset": Tensor[B, 2, H, W],
}
```

Semantic and center values are logits. The decoder applies sigmoid to the center output. Offset channels are `[dy, dx]` in pixels of the resized training image. `H` and `W` must be divisible by 16.

## Add the implementation

1. Put the model in `src/panoptic_segmenter/models/`.
2. Return the three named tensors at the input resolution.
3. Register a factory with `register_model()`.
4. Add a configuration file under `configs/`.
5. Add a CPU shape test and a backward test that does not download weights.
6. Run `train --dry-run` and one synthetic end-to-end test.

The factory receives `in_channels`, `num_classes`, and `base_channels`. If the model needs more settings, add explicit config fields and pass them through the factory; do not read hidden global state.

## What to document

State the input stride, parameter count, expected memory use, pretrained-weight behavior, and known failure modes. Include the output tensor shapes in the model page and config example.

The checkpoint stores the factory name and model settings. Changing output names, tensor meanings, or target units is a compatibility change. Add a migration or a new checkpoint schema rather than silently loading old weights with new semantics.

## Review checklist

- Output keys and shapes have tests.
- CPU forward and backward work offline.
- `Predictor` handles source images whose dimensions differ from the training size.
- Unsupported input sizes fail with a useful message.
- Model name, config fields, CLI examples, and both language versions agree.
- Performance claims include the device and input size used to measure them.
