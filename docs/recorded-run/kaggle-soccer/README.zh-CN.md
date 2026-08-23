# Kaggle Soccer 运行

[English](README.md) | [公共 Kaggle 指南](../../guides/kaggle-soccer.zh-CN.md) | [Kernel metadata](kernel-metadata.json)

## 状态：已提交

该 kernel 挂载公共 `quantigoai/soccer-dataset`，把视频/COCO 多边形标注转换为项目契约，生成 group-aware manifest，训练七类 Panoptic U-Net，并写出总体和每图评估产物。

该运行是教学证据，不是官方 benchmark。源数据集使用 CC-BY-SA-4.0，转换器使用 `frame_stride=5` 抽帧，train/valid/test 按源视频 group 切分。

## 复现

```bash
kaggle kernels push -p docs/recorded-run/kaggle-soccer
kaggle kernels status yashowhoo/pytorch-panoptic-segmentation-lab-kaggle-soccer
```

完成后下载：

```bash
kaggle kernels output yashowhoo/pytorch-panoptic-segmentation-lab-kaggle-soccer \
  --file-pattern 'artifacts/.*|kaggle-run-summary.json' -p kaggle-soccer-output
```

最终报告应保留源码 revision、转换参数、group split 数、resolved config、checkpoint hash、总体指标、per-class 指标和失败样本。
