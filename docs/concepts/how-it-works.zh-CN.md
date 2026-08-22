# 完整流程

[English](how-it-works.md) | [代码导览](code-tour.zh-CN.md)

```text
三个源目录
  -> 确定性 manifest + schema + identity
  -> 标签预检
  -> 同步 resize/flip
  -> semantic + Gaussian center + offset target
  -> PanopticUNet 三头输出
  -> 加权 loss 与 optimizer
  -> 有上限、类别约束的解码
  -> 按类别 PQ 累计
  -> 安全 checkpoint、指标与可视化
```

数据准备与训练刻意分离。manifest 固定 split 和 row 契约，schema 固定类别顺序、thing/stuff、颜色和 ignore ID。checkpoint 内嵌 resolved config 与 schema，评估不会从文件名猜测。

semantic 负责类别，center/offset 只解决 thing 身份。中心不能修复错误语义类别，完美 semantic 也不保证实例被正确分开。

后处理属于实验语义。center threshold、NMS、top-k 和最小面积都会改变 PQ，因此必须配置并保存，而不是隐藏参数。

validation 选择模型，test 只测一次选中的模型。运行环境和数据身份让结果可审计；即使流程完整，只有数据转换和 evaluator 符合具体 benchmark 协议时才是官方可比结果。
