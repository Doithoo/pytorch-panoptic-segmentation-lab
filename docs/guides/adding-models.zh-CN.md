# 添加模型

[English](adding-models.md) | [模型输出](../tutorial/03-panoptic-unet.zh-CN.md) | [代码导览](../concepts/code-tour.zh-CN.md)

训练代码要求模型返回三个张量：

```python
{
    "semantic": Tensor[B, C, H, W],
    "center": Tensor[B, 1, H, W],
    "offset": Tensor[B, 2, H, W],
}
```

semantic 和 center 是 logits，解码器会对 center 使用 sigmoid。offset 两个通道表示缩放后训练图像像素单位的 `[dy, dx]`。`H` 和 `W` 必须被 16 整除。

## 添加实现

1. 将模型放在 `src/panoptic_segmenter/models/` 下。
2. 在输入分辨率返回上面三个命名张量。
3. 使用 `register_model()` 注册构造函数。
4. 在 `configs/` 下添加配置文件。
5. 添加无需下载权重的 CPU shape 测试和 backward 测试。
6. 执行 `train --dry-run` 和一次合成数据端到端测试。

构造函数会收到 `in_channels`、`num_classes` 和 `base_channels`。如果模型需要其他参数，增加明确的配置字段并通过构造函数传入，不要读取隐藏的全局状态。

## 需要说明什么

记录输入 stride、参数量、预计显存、预训练权重行为和已知失败模式。在模型说明页和配置示例中写出输出张量的 shape。

checkpoint 会保存构造函数名称和模型设置。改变输出名称、张量含义或 target 单位都属于兼容性变更。应增加迁移方式或新的 checkpoint schema，不要让旧权重在新语义下静默加载。

## Review checklist

- 已测试输出 key 和 shape。
- CPU forward 和 backward 无需联网即可运行。
- `Predictor` 能处理与训练尺寸不同的源图像。
- 不支持的输入尺寸会给出有用的错误信息。
- 模型名、配置字段、CLI 示例和两种语言的文档一致。
- 性能数据注明测试设备和输入尺寸。
