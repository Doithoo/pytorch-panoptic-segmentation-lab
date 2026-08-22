# 03: Panoptic U-Net

[简体中文](03-panoptic-unet.zh-CN.md) | [Previous](02-data-and-targets.md) | [Next](04-training.md)

`PanopticUNet` is an educational baseline, not an implementation of the full Panoptic-DeepLab paper. It uses four encoder stages, a bottleneck, transposed-convolution upsampling, skip concatenation, and one shared final feature map.

Three 1x1 heads project that map:

- semantic: `num_classes` unnormalized logits;
- center: one unnormalized heatmap logit;
- offset: two unrestricted coordinate deltas.

Input height and width must be divisible by 16 because four pooling operations halve the grid. Batch dimension and spatial dimensions are preserved at the outputs.

```bash
uv run python examples/02_model_contract.py
```

The shared decoder keeps the model compact and makes the three-task contract visible. It also limits representational independence: semantic boundaries and instance localization may prefer different decoders. A stronger reference model could add a pretrained MobileNet/ResNet backbone, ASPP, separate semantic/instance decoders, deformable convolutions, or multi-scale context.

Such additions should enter through a real model registry/spec contract, save all constructor parameters, and retain CPU shape tests. Do not call a model “Panoptic-DeepLab” unless its architecture and training behavior match the cited design closely enough to defend that name.
