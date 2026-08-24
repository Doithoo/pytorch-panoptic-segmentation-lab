# ADR 0001：一个便于跟踪的轻量全景模型

[English](0001-readable-panoptic-baseline.md)

- 状态：已接受
- 决策日期：2026-08-22

## 要解决的问题

第一次做全景分割时，需要看清标签如何变成监督目标、模型的三个输出如何组合成实例，以及指标如何计算。大型框架可以减少代码量，却容易把这些步骤藏起来。项目还要能在 CPU 和 Kaggle GPU 上运行，同时避免解码器的显存随中心数量无界增长。

## 决策

使用带三个输出 head 的轻量 U-Net：

- semantic logits 负责类别预测；
- center heatmap 负责定位 thing 实例；
- 两个 offset 通道指向实例中心。

输入格式把 semantic mask 和 instance mask 分开保存。图像和 mask 使用完全相同的几何变换，然后在变换后的像素网格上生成 Gaussian center 和 offset。解码时只使用同类别中心，并限制中心数量和区域最小面积。PQ 在整个 split 上按类别累计。

格式 v1 不表达 crowd 区域。具体数据集的转换器和 evaluator 必须在这套基础格式之外处理 crowd 和 void 规则。

每次运行都保存最终配置、数据指纹、环境、随机数状态和 checkpoint 状态。这样重新查看结果时不需要依赖命令历史。

## 结果

代码足够小，可以用手工 Tensor 和 CPU 测试逐步跟踪。它不是完整的 Panoptic-DeepLab 实现，也不以高精度为目标。新增模型或数据集时，应保持输出 shape、数据规则、checkpoint 重建方式和指标测试不变。改变这些含义时，需要提供迁移方式或新的 schema version。
