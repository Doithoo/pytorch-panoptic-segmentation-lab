# 源码导览

[English](code-tour.md) | [一个样本如何经过整条流程](how-it-works.zh-CN.md)

先从 `cli.py` 看命令如何进入包。它只负责解析参数，具体工作由下面这些模块完成。

## 数据

- `data/schema.py`：类别 ID、颜色、thing/stuff 标记和 ignore 值。
- `data/manifest.py`：文件配对、随机或按组划分、哈希和数据指纹。
- `data/inspection.py`：打开 mask，检查取值和相互关系。
- `data/registry.py`：按名称查找数据转换器。
- `data/dataset.py`：读取一行 manifest 并组成 batch。
- `data/transforms.py`：同步缩放和翻转图像与 mask，然后生成监督目标。
- `data/soccer.py`：转换公开 Soccer 视频标注。

## 模型和训练

- `models/__init__.py`：模型 registry 和 factory 查找。
- `models/panoptic_unet.py`：encoder、decoder 以及 semantic/center/offset head。
- `training/losses.py`：语义交叉熵、center focal loss 和只作用于 thing 的 offset L1。
- `training/train.py`：loader、optimizer、scheduler、训练循环、验证和恢复。
- `training/checkpoint.py`：checkpoint 字段、安全加载、随机数状态和运行信息。

## 解码和结果

- `inference/postprocess.py`：中心筛选和同类别实例分配。
- `evaluation/metrics.py`：按类别累计 PQ/SQ/RQ。
- `evaluation/evaluate.py`：加载 checkpoint、检查数据指纹、计算总体指标和逐图报告。
- `evaluation/visualization.py`：semantic 配色和实例 overlay。
- `inference/predictor.py`：基于 checkpoint 预测并恢复原图尺寸。

名称以 `_` 开头的函数是内部辅助函数。包外代码应使用 CLI 或文档中列出的公共函数，不要直接依赖这些辅助函数。
