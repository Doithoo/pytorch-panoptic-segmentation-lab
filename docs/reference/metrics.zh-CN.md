# 指标参考

[English](metrics.md) | [评估教程](../tutorial/05-evaluation-and-inference.zh-CN.md)

每个类别内，预测与 target segment 的 IoU 严格大于 `0.5` 时匹配，每个 segment 最多匹配一次。

```text
PQ = 匹配 IoU 之和 / (TP + 0.5 FP + 0.5 FN)
SQ = 匹配 IoU 之和 / TP
RQ = TP / (TP + 0.5 FP + 0.5 FN)
```

统计先在整个 split 内按类别累计，再对有定义的类别宏平均。`pq_thing`、`pq_stuff` 分别平均有效 thing/stuff 类；`pq:class_<id>` 等字段给出类别值。完全缺席的类在 Python/YAML 中可能为 NaN，Kaggle JSON 会写 null。

未匹配预测若超过一半面积落在 target void，不计 FP；void 也从匹配 union 中排除。该实现修复了旧版无效 ignore 扣除和跨实例 micro 聚合。

限制：schema v1 没有 crowd 字段和数据集原生 segment JSON，因此适用于项目的 non-crowd 三 mask 契约。发布 Cityscapes/COCO 官方声明必须与其 evaluator 对拍并遵守服务器协议。

训练 `metrics.csv` 每轮报告 validation，`best.pt` 使用配置指标选择。最终 test 应只在选择后运行，并记录准确 checkpoint 哈希。
