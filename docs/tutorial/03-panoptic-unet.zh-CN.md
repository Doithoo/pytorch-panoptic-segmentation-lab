# 03：Panoptic U-Net

[English](03-panoptic-unet.md) | [上一节](02-data-and-targets.zh-CN.md) | [下一节](04-training.zh-CN.md)

`PanopticUNet` 是教学基线，不是 Panoptic-DeepLab 论文的完整实现。它包含四级 encoder、bottleneck、转置卷积上采样、skip 拼接，并共享最终 feature map。

三个 1x1 head 分别输出：

- semantic：`num_classes` 个未归一化 logits；
- center：一个未归一化热图 logit；
- offset：两个不受限的坐标增量。

四次 pooling 要求输入高宽可被 16 整除，三个输出保持 batch 和空间尺寸。

```bash
uv run python examples/02_model_contract.py
```

共享 decoder 让三任务契约清晰且模型紧凑，但也限制了语义边界与实例定位的独立表达。更强模型可加入预训练 MobileNet/ResNet、ASPP、独立 semantic/instance decoder、可变形卷积或多尺度上下文。

扩展模型应通过正式 registry/spec 契约接入，保存全部构造参数，并保留 CPU shape 测试。除非架构和训练行为足够接近论文设计，否则不要把模型直接命名为“Panoptic-DeepLab”。
