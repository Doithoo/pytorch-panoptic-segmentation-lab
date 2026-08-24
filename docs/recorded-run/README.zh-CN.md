# Kaggle 参考运行

[English](README.md) | [Kaggle 指南](../guides/kaggle.zh-CN.md) | [参考配置](../../configs/reference_kaggle.yaml) | [Kaggle 页面](https://www.kaggle.com/code/yashowhoo/pytorch-panoptic-segmentation-lab-gpu)

## 状态：完成

Kaggle version 2 在固定源码 revision 和 Tesla T4 上运行了仓库。它生成 256 张合成图像，准备数据清单，训练 20 轮，重新加载 `best.pt`，评估 test，并导出下面列出的文件。

数据由仓库生成，因此这些数字只说明合成任务和项目的 non-crowd mask 规则，不代表 Cityscapes 或 COCO 结果。

## 结果

| 项目 | 值 |
|---|---:|
| Kaggle 版本 | 2 |
| 硬件 | Tesla T4（分配 2 张卡，实际使用 1 张） |
| Python / PyTorch | 3.12.13 / 2.10.0+cu128 |
| 源码 revision | `f6fb554066d508f933fe220bf27c39ad2de04d8c` |
| 数据 | 256 张生成图像，源尺寸 128x128 |
| 划分 | train 205 / valid 26 / test 25 |
| 模型 | Panoptic U-Net，base channels 32 |
| 训练 | 20 轮、AdamW、cosine、CUDA AMP |
| 最优 validation PQ | **0.865159**，第 20 轮 |
| Test PQ | **0.853881** |
| Test SQ / RQ | 0.973944 / 0.869757 |
| Test PQ thing / stuff | **0.561655 / 0.999993** |
| Test TP / FP / FN | 96 / 55 / 4 |
| 耗时 | 83.762 秒 |
| 最优 checkpoint SHA-256 | `30f7905f84fef4db783b6bca7185a1520f2fa5247fc80c23db7ca03c4d32a43a` |

在这组合成数据中，stuff 比对象分离更容易学会。修改 center target、offset loss 或解码阈值时，应同时查看按类别指标和 overlay。

## 文件

| 文件 | 内容 |
|---|---|
| [`kaggle-run-summary.json`](kaggle-run-summary.json) | 状态、设备、版本、划分、指标和 checkpoint hash |
| [`reference-panoptic-unet/config.yaml`](reference-panoptic-unet/config.yaml) | 实际训练和后处理参数 |
| [`reference-panoptic-unet/run.yaml`](reference-panoptic-unet/run.yaml) | 环境、数据指纹、seed、耗时和选择指标 |
| [`reference-panoptic-unet/metrics.csv`](reference-panoptic-unet/metrics.csv) | 全部训练和验证记录 |
| [`reference-panoptic-unet/evaluation/evaluation.json`](reference-panoptic-unet/evaluation/evaluation.json) | Test 总体指标 |
| [`reference-panoptic-unet/evaluation/per_class.csv`](reference-panoptic-unet/evaluation/per_class.csv) | 每类 PQ/SQ/RQ |

checkpoint 不保存在 Git 中。需要查看 tensor 时，从 Kaggle 下载：

```bash
kaggle kernels output yashowhoo/pytorch-panoptic-segmentation-lab-gpu \
  --file-pattern 'artifacts/reference-panoptic-unet/best.pt' -p kaggle-output
```

## 复现

提交的文件是 [`kaggle/run_kaggle.py`](kaggle/run_kaggle.py) 和 [`kernel-metadata.json`](kaggle/kernel-metadata.json)。提交新运行前，请更新其中固定的源码 revision。
