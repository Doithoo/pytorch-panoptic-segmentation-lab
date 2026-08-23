# Kaggle Soccer 运行

[English](README.md) | [公共 Kaggle 指南](../../guides/kaggle-soccer.zh-CN.md) | [Kernel 页面](https://www.kaggle.com/code/yashowhoo/pytorch-panoptic-segmentation-lab-kaggle-soccer)

## 状态：完成

Kaggle kernel version 2 完成了完整的公开数据教学流程：源码 checkout、Soccer 视频/COCO 多边形转换、group-aware manifest 准备、CUDA preflight、训练、checkpoint 重载、test 评估和详细失败指标。

这是教学证据，不是官方 benchmark。源数据集使用 CC-BY-SA-4.0，转换器使用 `frame_stride=5` 抽帧，train/valid/test 按源视频 group 切分。

## 结果

| 项目 | 值 |
|---|---:|
| Kaggle 版本 | 2 |
| 硬件 | Tesla T4 |
| Python / PyTorch | 3.12.13 / 2.10.0+cu128 |
| 源码 revision | `e88a11d488ad0f02f476f7143c76484b73ed579b` |
| 转换参数 | 最大 240 帧、stride 5、宽度 512 |
| Split | train 85 / valid 46 / test 39 |
| Groups | train `Batch 3`、valid `Batch 1`、test `Batch 2` |
| 模型 | Panoptic U-Net，base channels 16 |
| 训练 | 10 epoch、AdamW、cosine、CUDA AMP |
| 最优 validation PQ | **0.290397**，第 6 epoch |
| Test PQ | **0.223444** |
| Test SQ / RQ | 0.878695 / 0.251701 |
| Test PQ thing / stuff | **0.000000 / 0.391027** |
| Test TP / FP / FN | 63 / 5618 / 266 |
| 总耗时 | 149.5 秒 |
| 最优 checkpoint SHA-256 | `3e0ea31a7f1482702752beea285f6fdd8c27b5bb7468f35e92a582e3ca4f2d08` |

较低的 thing 分数和较高的 false positive 数量本身就是有价值的教学证据：当前从头训练的小模型能学习大范围 stuff 区域，但还不能可靠地区分球员、足球和裁判。这正是 center/offset target、后处理阈值和逐图报告需要帮助用户理解的失败类型。

## 证据文件

| 文件 | 内容 |
|---|---|
| [`kaggle-run-summary.json`](kaggle-run-summary.json) | GPU、源码 revision、转换参数、split、指标和 checkpoint hash |
| [`reference-kaggle-soccer/config.yaml`](reference-kaggle-soccer/config.yaml) | 七类训练的 resolved 配置 |
| [`reference-kaggle-soccer/dataset.yaml`](reference-kaggle-soccer/dataset.yaml) | 数据 identity 和 group-aware split 证据 |
| [`reference-kaggle-soccer/run.yaml`](reference-kaggle-soccer/run.yaml) | 环境、设备、seed 和时间元数据 |
| [`reference-kaggle-soccer/metrics.csv`](reference-kaggle-soccer/metrics.csv) | 十轮训练/验证记录 |
| [`reference-kaggle-soccer/evaluation/evaluation.json`](reference-kaggle-soccer/evaluation/evaluation.json) | Test 总体指标 |
| [`reference-kaggle-soccer/evaluation/evaluation_detailed.json`](reference-kaggle-soccer/evaluation/evaluation_detailed.json) | 每图指标和 worst cases |
| [`reference-kaggle-soccer/evaluation/per_class.csv`](reference-kaggle-soccer/evaluation/per_class.csv) | 每类 PQ/SQ/RQ |

checkpoint 不提交到仓库，需要时从 Kaggle output 下载：

```bash
kaggle kernels output yashowhoo/pytorch-panoptic-segmentation-lab-kaggle-soccer \
  --file-pattern 'artifacts/kaggle-soccer-panoptic-unet/best.pt' -p kaggle-soccer-output
```

## 复现

Kernel 文件是 [`run_kaggle.py`](run_kaggle.py) 和 [`kernel-metadata.json`](kernel-metadata.json)。runner 固定已审查的源码 revision，并挂载 `quantigoai/soccer-dataset`。

```bash
kaggle kernels push -p docs/recorded-run/kaggle-soccer
kaggle kernels status yashowhoo/pytorch-panoptic-segmentation-lab-kaggle-soccer
```

## 限制

本运行使用小型公开视频数据集，不是 Cityscapes 或 COCO。group-aware split 防止相邻视频泄漏，但源数据集没有官方 benchmark 协议，也没有项目特定的 crowd policy。该结果不能直接与 leaderboard 分数比较。
