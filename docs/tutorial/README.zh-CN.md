# 全景分割教程

[English](README.md) | [文档中心](../README.zh-CN.md) | [项目首页](../../README.zh-CN.md)

项目首页提供最短可运行流程；本页按“我想回答什么问题”选择教程章节。

| 章节 | 回答的问题 | 适合何时阅读 |
|---|---|---|
| [00 全景 Tensor 与 ID](00-basics.zh-CN.md) | semantic、instance、panoptic 分别是什么？ | 不熟悉 Tensor shape 或 ID |
| [01 环境和 CLI](01-environment.zh-CN.md) | 如何安装并选择设备？ | 命令无法运行时 |
| [02 数据与监督目标](02-data-and-targets.zh-CN.md) | mask 如何变成 center 和 offset target？ | 准备或替换数据时 |
| [03 Panoptic U-Net](03-panoptic-unet.zh-CN.md) | 三个模型 head 分别预测什么？ | 阅读模型代码时 |
| [04 训练与恢复](04-training.zh-CN.md) | loss、验证、checkpoint 和 resume 如何连接？ | 开始正式实验前 |
| [05 评估与推理](05-evaluation-and-inference.zh-CN.md) | 如何理解 PQ 和预测失败？ | 训练完成后 |

建议顺序为 `00 -> 02 -> 03 -> 04 -> 05`；第 01 章作为环境参考页随时查阅。每章都应以一个可以运行、检查或从源码解释的结果结束。

需要操作步骤时看[常见任务](../README.zh-CN.md#常见任务)，需要准确字段和公式时看[参考资料](../README.zh-CN.md#参考资料)。
