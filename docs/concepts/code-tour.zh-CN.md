# 代码导览

[English](code-tour.md) | [完整流程](how-it-works.zh-CN.md)

从 `cli.py` 开始，它只负责参数解析。`config.py` 合并默认值、严格 YAML 和带类型的 CLI override，并检查字段范围。

数据模块：

- `data/schema.py`：不可变类别和 thing/stuff 语义。
- `data/manifest.py`：确定性配对、split、哈希和 identity。
- `data/inspection.py`：解码后的标签完整性。
- `data/dataset.py`：row 加载与 batch。
- `data/transforms.py`：同步几何与训练 target。

模型与优化：

- `models/panoptic_unet.py`：共享 encoder/decoder 和三头。
- `training/losses.py`：semantic CE、center focal、thing-only offset L1。
- `training/train.py`：loader、optimizer/scheduler、训练评估与恢复。
- `training/checkpoint.py`：安全原子持久化、RNG、环境和哈希。

结果模块：

- `inference/postprocess.py`：有上限的中心提取和解码。
- `evaluation/metrics.py`：split 级按类别 PQ。
- `evaluation/evaluate.py`：checkpoint 评估。
- `evaluation/visualization.py`：schema 配色与实例 overlay。
- `inference/predictor.py`：保存尺度预处理、重载和原尺寸导出。

`_` 开头 helper 只供内部共享，不是稳定公共 API。扩展时应增加明确契约，不要让应用代码依赖深层私有函数。
