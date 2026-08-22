# 05：评估与推理

[English](05-evaluation-and-inference.md) | [上一节](04-training.zh-CN.md) | [指标参考](../reference/metrics.zh-CN.md)

评估会安全重建保存的架构、加载 tensor state、应用保存的后处理配置，并评估 prepared split。`max_test_samples` 与 valid 上限相互独立。

解码器只在预测 thing 像素上执行 center sigmoid、NMS、阈值和全局 top-k，并分块把像素分配给同类别中心。小于 `instance_area` 的 thing、小于 `stuff_area` 的 stuff、没有中心的 thing 会变为 void，从而限制原先 `H x W x 全部中心` 的显存风险。

PQ 使用 IoU 大于 0.5 的匹配，先在整个 split 内按类别累计，再做宏平均。超过一半面积落在 target void 的预测不计 FP。输出总体 PQ/SQ/RQ、thing/stuff PQ 和各类别值。

本项目契约不含 crowd。与官方数据集比较时必须补充该数据集的 crowd/void 转换，并与官方 evaluator 对拍；没有适配时不要直接和 leaderboard 数值比较。

预测会 pad 到 16 的倍数，再裁剪回原图尺寸，输出 semantic ID、16-bit instance ID、稳定 schema 配色和实例 overlay。阈值来自 checkpoint 配置。

聚合指标回答实验是否改善，可视化回答如何失败。应分别检查语义混淆、中心漏检/重复、实例合并/拆分、小区域过滤和 offset 方向。
