# 文档目录

[English](README.md) | [项目首页](../README.zh-CN.md)

想立即运行代码，请先看项目首页。遇到具体问题时，可以从下面的表格直接进入相应教程或操作指南。

## 第一次运行

| 目标 | 文档 |
|---|---|
| 在 CPU 上完成最短流程 | [学习路线](tutorial/learning-path.zh-CN.md) |
| 按问题选择一章教程 | [教程目录](tutorial/README.zh-CN.md) |
| 转换并训练公开 Soccer 数据 | [Kaggle Soccer](guides/kaggle-soccer.zh-CN.md) |
| 查看已完成运行的真实输出 | [合成数据运行](recorded-run/README.zh-CN.md)和 [Soccer 运行](recorded-run/kaggle-soccer/README.zh-CN.md) |

## 教程

| 问题 | 章节 |
|---|---|
| semantic 和 instance ID 如何表示？ | [全景分割张量与 ID](tutorial/00-basics.zh-CN.md) |
| 如何安装项目和选择设备？ | [环境和 CLI](tutorial/01-environment.zh-CN.md) |
| mask 如何生成 center 和 offset 监督？ | [数据与监督目标](tutorial/02-data-and-targets.zh-CN.md) |
| 模型的三个输出分别表示什么？ | [Panoptic U-Net](tutorial/03-panoptic-unet.zh-CN.md) |
| 训练和断点恢复执行了什么？ | [训练与恢复](tutorial/04-training.zh-CN.md) |
| PQ 和预测文件如何生成？ | [评估与推理](tutorial/05-evaluation-and-inference.zh-CN.md) |

## 常见任务

| 任务 | 指南 |
|---|---|
| 使用公开 Soccer 数据集 | [Kaggle Soccer](guides/kaggle-soccer.zh-CN.md) |
| 在 Kaggle 上运行合成数据 | [Kaggle GPU](guides/kaggle.zh-CN.md) |
| 转换 Cityscapes | [Cityscapes](guides/cityscapes.zh-CN.md) |
| 准备其他数据集 | [使用自己的数据](guides/using-your-data.zh-CN.md) |
| 编写数据集转换器 | [添加数据集](guides/adding-datasets.zh-CN.md) |
| 注册其他模型 | [添加模型](guides/adding-models.zh-CN.md) |
| 比较多次运行 | [实验对比](guides/experiments.zh-CN.md) |
| 排查报错或低质量结果 | [故障排查](guides/troubleshooting.zh-CN.md) |

## 理解代码

- [一个样本如何经过整条流程](concepts/how-it-works.zh-CN.md)
- [源码导览](concepts/code-tour.zh-CN.md)
- [配置值如何合并](concepts/configuration-flow.zh-CN.md)
- [为什么模型输出 semantic、center 和 offset](architecture/0001-readable-panoptic-baseline.zh-CN.md)

## 参考资料

需要查询准确字段、文件结构或公式时使用：

- [CLI 命令](reference/cli.zh-CN.md)
- [配置字段](reference/config-reference.zh-CN.md)
- [数据和 manifest 格式](reference/data-format.zh-CN.md)
- [PQ、SQ 和 RQ](reference/metrics.zh-CN.md)
- [Checkpoint 字段和恢复规则](reference/checkpoint-schema.zh-CN.md)
- [Cityscapes 类别映射](reference/cityscapes.zh-CN.md)
- [运行环境和产物兼容性](reference/compatibility.zh-CN.md)

教程解释原理，指南给出操作步骤，参考资料定义准确行为，运行记录保存固定命令产生的实测结果。
