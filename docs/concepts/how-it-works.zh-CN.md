# 一个样本如何经过整条流程

[English](how-it-works.md) | [代码导览](code-tour.zh-CN.md)

```text
图像 + 语义 mask + 实例 mask
  -> 固定的 train/valid/test 数据清单
  -> 文件和标签检查
  -> 同步缩放与水平翻转
  -> 语义、中心热图和 offset 监督
  -> 模型的三个输出
  -> 加权损失和参数更新
  -> 中心检测与实例分配
  -> PQ/SQ/RQ 计算
  -> checkpoint、训练指标和预测图
```

## 训练之前

`prepare-data` 按文件名配对图像和两张 mask，并生成 CSV 数据清单。之后的命令都使用这份固定划分。视频帧或相邻裁剪之间存在关联时，可以通过 group 文件让它们留在同一个数据集中。

`schema.yaml` 定义类别顺序、显示颜色、忽略值，以及每个类别属于 thing 还是 stuff。随后，`inspect-data` 会打开文件，检查尺寸、标签取值、实例 ID、文件哈希和数据划分。

## 训练期间

图像和两张 mask 会经过完全相同的缩放与翻转。监督目标在变换后生成，因此中心坐标和 offset 都以缩放后的像素网格为准。

模型有三个输出：

- semantic logits 判断每个像素属于哪个类别；
- center heatmap 定位 thing 实例的中心；
- offset 指向每个 thing 像素所属实例的中心。

语义分类和实例分离解决的是两个不同问题。语义类别预测正确，不代表相邻的两个人一定能分开；中心位置预测正确，也无法修复错误的语义类别。

## 前向计算之后

训练损失由语义交叉熵、中心 focal loss 和只作用于 thing 像素的 offset L1 组成。每个 epoch 结束后在验证集上评估，并按照 `train.best_metric` 选择 `best.pt`。

评估和预测时，解码器先寻找中心峰值，再把 thing 像素分配给同类别中心，过滤面积过小的区域；stuff 像素的 instance ID 保持为 0。中心阈值、NMS 大小、top-k 和面积阈值都会影响 PQ，因此它们随 checkpoint 一起保存。

模型和阈值在验证集上确定后，再评估测试集。通过 `run.yaml`、`config.yaml`、`dataset.yaml` 和 checkpoint 哈希，可以查清一个结果使用了哪版代码、哪份数据划分和哪些参数。
