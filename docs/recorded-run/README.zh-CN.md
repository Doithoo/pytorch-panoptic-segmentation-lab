# Kaggle 参考运行状态

[English](README.md) | [Kaggle 指南](../guides/kaggle.zh-CN.md) | [参考配置](../../configs/reference_kaggle.yaml)

## 状态：等待实际执行

可提交 kernel 和产物契约已经完成，但仓库尚未取回一次成功 Kaggle GPU 输出。本页不会虚构指标、耗时、硬件分配或 Notebook URL。

待执行协议：

| 项目 | 固定值 |
|---|---|
| 数据 | 256 张确定性合成图，源尺寸 128x128 |
| Split | seed 42，0.8 / 0.1 / 0.1 |
| 模型 | Panoptic U-Net，base channels 32 |
| 训练 | 20 epoch、AdamW、cosine、CUDA AMP |
| 输入 | resize 256x256，batch 4，workers 2 |
| 选择 | validation PQ 最优 |
| 最终评估 | 从 `best.pt` 加载并评估 test |
| 硬件 | Kaggle T4 或更新 |

提交文件为 `kaggle/run_kaggle.py` 与 `kaggle/kernel-metadata.json`，runner 会记录实际 source commit 和最终 checkpoint 哈希。

## 完成验收

只有满足以下条件才能替换 pending：Kaggle 状态为 `COMPLETE`；20 行指标完整且有限；`best.pt` 能安全重载并评估 test；保留 summary/config/run/metrics/evaluation；声明 revision、checkpoint SHA-256、GPU、版本、耗时和 split；链接 Kaggle 页面；明确合成结果只是流程证据。

不要提交大型 checkpoint，只保留小型 CSV/YAML/JSON 和精选可视化。真实数据记录还必须说明 converter、许可、官方 split、crowd/void evaluator 兼容和数据身份。
