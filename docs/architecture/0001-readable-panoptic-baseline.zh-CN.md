# ADR 0001：可读的全景分割基线

[English](0001-readable-panoptic-baseline.md)

- 状态：已接受
- 决策日期：2026-08-22

## 背景

项目需要在不依赖大型框架隐藏 target 契约的前提下，讲清语义与实例推理，并让 CPU/Kaggle 上的评估和产物可审计。

## 决策

采用 semantic、center、offset 三头轻量 U-Net；源标签使用独立 semantic/instance mask；同步几何后构建 Gaussian center；按同类别、中心数量受限的方式解码；在整个 split 内按类别累计 PQ。schema v1 不包含 crowd，并明确声明限制。

采用严格 dataclass 配置、prepared-data identity、预检和带版本的 `weights_only=True` checkpoint。后处理属于保存的实验语义。先发布确定性合成 Kaggle 运行，再声明任何真实 benchmark。

## 影响

实现可读，能用手工 tensor 测试；但它不等价于完整 Panoptic-DeepLab，没有 adapter 时也不能报告 crowd-aware Cityscapes/COCO 官方分数。未来 model registry 或 crowd schema 必须保持 checkpoint 重建和指标测试，必要时引入新的明确 schema version。
