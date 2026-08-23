# Adding A Model

[简体中文](adding-models.zh-CN.md) | [Model contract](../tutorial/03-panoptic-unet.md) | [Code tour](../concepts/code-tour.md)

The current training loop expects one model with three output heads:

```python
{
    "semantic": Tensor[B, C, H, W],
    "center": Tensor[B, 1, H, W],
    "offset": Tensor[B, 2, H, W],
}
```

`H` and `W` must match the transformed training image and be divisible by 16. Semantic values are logits, center values are logits passed through sigmoid by the decoder, and offset values are `[dy, dx]` predictions in resized-pixel units.

## Steps

1. Add the model implementation under `src/panoptic_segmenter/models/`.
2. Keep the forward contract explicit and add a CPU shape test plus a backward test with no downloaded weights.
3. Register the name with `register_model()` in the model registry.
4. Add a complete YAML example under `configs/`.
5. Document parameter count, input stride, pretrained-weight behavior, and limitations.
6. Run a dry-run and end-to-end synthetic training test.

The checkpoint stores the model name in the resolved config. Changing a model contract requires a compatibility decision and usually a new checkpoint schema or a clear migration rule. Do not load arbitrary import paths from a checkpoint; model reconstruction is a code trust boundary.

## Review checklist

- Output keys and shapes are tested.
- CPU forward/backward works offline.
- Odd source image sizes still work through `Predictor`.
- The model rejects unsupported input sizes with a clear error.
- The model name, config fields, README, CLI reference, and Chinese counterpart agree.
- Parameter and speed claims include hardware and input dimensions.
