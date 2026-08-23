# 添加模型

[English](adding-models.md) | [模型契约](../tutorial/03-panoptic-unet.zh-CN.md) | [代码导览](../concepts/code-tour.zh-CN.md)

当前训练循环要求模型提供三个输出 head：

```python
{
    "semantic": Tensor[B, C, H, W],
    "center": Tensor[B, 1, H, W],
    "offset": Tensor[B, 2, H, W],
}
```

`H` 和 `W` 必须与变换后的训练图像一致，并被 16 整除。semantic 是 logits，center 会在 decoder 中经过 sigmoid，offset 是以 resize 后像素为单位的 `[dy, dx]` 预测。

## 步骤

1. 在 `src/panoptic_segmenter/models/` 下实现模型。
2. 保持 forward 契约明确，加入无需下载权重的 CPU shape 和 backward 测试。
3. 使用 `register_model()` 在模型 registry 中注册名称。
4. 在 `configs/` 下增加完整 YAML 示例。
5. 记录参数量、输入 stride、预训练权重行为和限制。
6. 执行 dry-run 和合成数据端到端训练测试。

checkpoint 会在 resolved config 中保存模型名称。改变模型契约时必须做兼容性决策，通常需要新的 checkpoint schema 或明确的迁移规则。不要从 checkpoint 加载任意 import path；模型重建属于代码信任边界。

## Review checklist

- 已测试输出 key 和 shape。
- CPU forward/backward 无需联网即可运行。
- `Predictor` 能处理奇数尺寸源图像。
- 不支持的输入尺寸会给出清晰错误。
- 模型名、配置字段、README、CLI 参考和中文文档一致。
- 参数量和速度声明包含硬件与输入尺寸。
