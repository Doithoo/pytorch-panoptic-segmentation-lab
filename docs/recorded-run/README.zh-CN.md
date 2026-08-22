# Kaggle 参考运行

[English](README.md) | [Kaggle 指南](../guides/kaggle.zh-CN.md) | [参考配置](../../configs/reference_kaggle.yaml) | [Kaggle 页面](https://www.kaggle.com/code/yashowhoo/pytorch-panoptic-segmentation-lab-gpu)

## 状态：已完成

确定性合成数据参考任务已在 Kaggle version 2 成功完成。它在一次非交互任务中验证源码 checkout、package 安装、CUDA 执行、数据准备、训练、checkpoint 重载、test 评估和产物导出。

这只是流程证据，不是真实世界 benchmark。数据由仓库生成，指标遵守项目的 non-crowd 三 mask 契约，不能表述为 Cityscapes 或 COCO 性能。

## 结果

| 项目 | 值 |
|---|---:|
| Kaggle version | 2 |
| 硬件 | Tesla T4（分配 2 张，项目使用 1 张） |
| Python / PyTorch | 3.12.13 / 2.10.0+cu128 |
| 源码 revision | `f6fb554066d508f933fe220bf27c39ad2de04d8c` |
| 数据 | 256 张确定性合成图，源尺寸 128x128 |
| Split | train 205 / valid 26 / test 25 |
| 模型 | Panoptic U-Net，base channels 32 |
| 训练 | 20 epoch、AdamW、cosine、CUDA AMP |
| 最佳 validation PQ | **0.865159**，epoch 20 |
| Test PQ | **0.853881** |
| Test SQ / RQ | 0.973944 / 0.869757 |
| Test PQ thing / stuff | **0.561655 / 0.999993** |
| Test loss | 0.013969 |
| Test TP / FP / FN | 96 / 55 / 4 |
| 训练任务耗时 | 83.762 秒 |
| 最佳 checkpoint SHA-256 | `30f7905f84fef4db783b6bca7185a1520f2fa5247fc80c23db7ca03c4d32a43a` |

stuff 分数很高而 thing 分数较低符合教学数据特征：大块背景更容易学习，对象分离更难。该结果用于检查流程和错误分解，不用于跨数据集比较模型质量。

## 证据文件

| 文件 | 内容 |
|---|---|
| [`kaggle-run-summary.json`](kaggle-run-summary.json) | 状态、GPU、版本、split、test 摘要、checkpoint 哈希 |
| [`reference-panoptic-unet/config.yaml`](reference-panoptic-unet/config.yaml) | 实际训练和后处理配置 |
| [`reference-panoptic-unet/run.yaml`](reference-panoptic-unet/run.yaml) | 环境、数据身份、seed、时间和 best 指标 |
| [`reference-panoptic-unet/metrics.csv`](reference-panoptic-unet/metrics.csv) | 全部 20 行训练/validation 指标 |
| [`reference-panoptic-unet/evaluation/evaluation.json`](reference-panoptic-unet/evaluation/evaluation.json) | test 聚合指标 |
| [`reference-panoptic-unet/evaluation/per_class.csv`](reference-panoptic-unet/evaluation/per_class.csv) | 类别级 PQ/SQ/RQ |

checkpoint 不提交到仓库，需要时从 Kaggle 下载：

```bash
kaggle kernels output yashowhoo/pytorch-panoptic-segmentation-lab-gpu \
  --file-pattern 'artifacts/reference-panoptic-unet/best.pt' -p kaggle-output
```

## 复现

提交文件是 [`kaggle/run_kaggle.py`](kaggle/run_kaggle.py) 和 [`kernel-metadata.json`](kaggle/kernel-metadata.json)，runner 固定到上面的源码 revision。创建新运行前应有意识地更新 revision 和 metadata，然后按 [Kaggle 指南](../guides/kaggle.zh-CN.md)执行。

## 限制

本次运行没有外部数据、crowd 标注或官方服务器 evaluator。真实 Cityscapes/COCO 结果仍需 converter、官方 split、crowd/void adapter、许可审查和数据集专用 evaluator 对拍。
