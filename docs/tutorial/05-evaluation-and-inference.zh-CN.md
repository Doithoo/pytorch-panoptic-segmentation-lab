# 05：评估与推理

[English](05-evaluation-and-inference.md) | [上一节](04-training.zh-CN.md) | [指标参考](../reference/metrics.zh-CN.md)

评估首先检查数据清单，并比较 `dataset.yaml` 的数据指纹和 checkpoint 中保存的指纹。之后才会重建保存的架构、加载模型状态、使用保存的后处理配置并评估指定 split。`max_test_samples` 与 valid 上限相互独立。

解码器只在预测 thing 像素上执行 center sigmoid、NMS、阈值和全局 top-k，并分块把像素分配给同类别中心。小于 `instance_area` 的 thing、小于 `stuff_area` 的 stuff、没有中心的 thing 会变为 void，从而限制原先 `H x W x 全部中心` 的显存风险。

PQ 使用 IoU 大于 0.5 的匹配，先在整个 split 内按类别累计，再做宏平均。超过一半面积落在 target void 的预测不计 FP。输出总体 PQ/SQ/RQ、thing/stuff PQ 和各类别值。

项目使用的格式不包含 crowd 区域。如果要和某个数据集公开的分数比较，必须按照该数据集的规则转换标签和预测结果，再运行它自己的 evaluator。内置 evaluator 只适用于本项目的 non-crowd mask。

预测先把输入缩放到 checkpoint 保存的训练 `data.image_size`，在该尺度解码，再用 nearest-neighbor 把离散 mask 恢复到原图尺寸。输出 semantic ID、16-bit instance ID、稳定 schema 配色和实例 overlay，阈值来自 checkpoint 配置。

CLI 可以把 JSON 报告写到运行目录旁边：

```bash
uv run panoptic-segment evaluate artifacts/run/best.pt --split test \
  --output artifacts/run/evaluation.json
```

报告包含 checkpoint SHA-256、请求与解析后的设备、split、数据指纹、总体指标、每图指标和最低 PQ 列表。

聚合指标回答实验是否改善，可视化回答如何失败。应分别检查语义混淆、中心漏检/重复、实例合并/拆分、小区域过滤和 offset 方向。
