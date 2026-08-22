# 00: Panoptic Tensors and IDs

[简体中文](00-basics.zh-CN.md) | [Learning path](learning-path.md)

Panoptic segmentation answers two questions for every pixel: what class is it, and for thing classes, which object does it belong to?

The project stores ground truth as:

- `semantic[y,x]`: contiguous class ID or `255` for void.
- `instance[y,x]`: positive image-local object ID for thing pixels, otherwise `0`.

The instance ID has no class meaning by itself. `(semantic=1, instance=7)` identifies one region; ID 7 in another image is unrelated. A thing instance may not span semantic classes.

The model returns:

```text
semantic logits [B,C,H,W]
center logits   [B,1,H,W]
offset          [B,2,H,W]
```

Cross-entropy learns semantic classes. A Gaussian heatmap makes each object center a learnable local peak. At each thing pixel `(y,x)`, offset target `[dy,dx]` points to the instance mean `(cy,cx)`: `dy=cy-y`, `dx=cx-x`.

During decoding, semantic `argmax` identifies thing pixels. Center NMS finds bounded peaks. Each thing pixel adds its predicted offset and joins the nearest center of the same class. Stuff needs no instance ID.

Do not train cross-entropy on `argmax` values, resize labels with bilinear interpolation, assign instance IDs across images, or interpret a semantic mask alone as panoptic output.
