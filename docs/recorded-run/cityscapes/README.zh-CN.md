# Cityscapes Kaggle 运行状态

[English](README.md) | [Cityscapes 指南](../../guides/cityscapes.zh-CN.md)

## 状态：等待挂载有许可的私有数据集

转换、训练、预测导出、官方 ground truth 生成和 `cityscapesscripts` 评估 runner 均已实现。这里没有记录真实 Cityscapes 指标，因为仓库和当前 Kaggle 账号中都没有同时包含 `leftImg8bit` 与 `gtFine instanceIds` 的官方许可归档。

Kaggle 公开搜索结果通常是 pix2pix 图像对或 semantic mask，不是完整官方 panoptic 源目录。项目不会使用未经验证的重新上传数据来制造 benchmark 结果。

完成步骤：

1. 按官方许可从 Cityscapes 门户下载 `leftImg8bit_trainvaltest.zip` 和 `gtFine_trainvaltest.zip`。
2. 将压缩包或解压目录上传为不再分发的私有 Kaggle Dataset。
3. 把该 Dataset 挂载到 [`kaggle/kernel-metadata.json`](kaggle/kernel-metadata.json) 描述的私有 kernel。
4. 把 [`kaggle/run_kaggle.py`](kaggle/run_kaggle.py) 固定到审查后的仓库 commit。
5. 提交任务并保留 summary、resolved config、metrics、官方 validation JSON、checkpoint 哈希和错误可视化。

runner 会发现唯一包含 `leftImg8bit/` 和 `gtFine/` 的根目录，保留官方 train/val，在 train 训练、val 选择，按保存尺度推理并恢复全尺寸 mask，生成保留 crowd 的官方 ground truth，再通过 `cityscapesscripts` 评估。它不会把 val 称为 test。
