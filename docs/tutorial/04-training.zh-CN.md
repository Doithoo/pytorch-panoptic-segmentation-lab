# 04：训练、产物与恢复

[English](04-training.md) | [上一节](03-panoptic-unet.zh-CN.md) | [下一节](05-evaluation-and-inference.zh-CN.md)

训练目标为：

```text
semantic_weight * cross_entropy
+ center_weight * center_focal
+ offset_weight * thing_pixel_L1
```

center focal 按真实中心数归一化并降低 Gaussian 边缘权重，避免大量背景支配训练；offset L1 只观察 thing 像素。总 loss 必须有限，`grad_clip > 0` 时裁剪梯度。

已实现 `adamw`、`sgd`，scheduler 支持 `none`、`step`、`cosine`。CUDA AMP 使用 GradScaler，CPU/MPS 使用全精度。每个 epoch 记录学习率和各项 loss。

每轮后执行 validation，由 `train.best_metric` 选择 `best.pt`，可选 PQ、SQ、RQ、thing PQ、stuff PQ。test 不参与训练。

Checkpoint schema v1 保存 model、optimizer、scheduler、scaler、epoch、best、完整历史、RNG、配置、schema、环境和数据身份。保存使用临时文件加 `os.replace`，加载使用 `weights_only=True`。

恢复允许增加总 epoch 以及改变数据路径、worker、device，但拒绝不同模型、loss、后处理、schema、数据身份、optimizer、scheduler、seed 和增强语义。恢复必须使用运行目录的 `last.pt`，`metrics.csv` 追加而非重写。

Dry run 会真实更新一次参数，但不写正常运行产物。它用于发现 shape、显存和非有限 loss 问题，不用于估计质量。
