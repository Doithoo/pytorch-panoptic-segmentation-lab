# 兼容性策略

[English](compatibility.md) | [Checkpoint schema](checkpoint-schema.zh-CN.md) | [配置字段](config-reference.zh-CN.md)

## 运行环境

支持 Python 3.10 至 3.12。`pyproject.toml` 为 PyTorch、torchvision、NumPy、Pillow 和 PyYAML 声明兼容版本范围；`uv.lock` 是可复现的开发环境。CI 会在三个 Python 版本上检查 CPU 流程。

CUDA 和 MPS 是可选设备。Kaggle 参考流程在 T4 级别 CUDA 环境测试过。CPU 测试通过不代表特定 CUDA build 一定包含目标 GPU 架构所需的 kernel。

## 持久化产物

- prepared data 格式：version 1。
- checkpoint schema：version 1。
- 模型名称通过保存配置中的 registry factory name 解析。
- 评估报告使用 JSON，缺席类别的指标可能为 `null`。

只有 schema、data identity、模型、loss、后处理和训练语义匹配时，checkpoint 才能恢复。当前 loader 遇到未知 checkpoint schema version 会拒绝加载，不会猜测迁移方式。

## 变更策略

涉及标签、target 字段、模型输出 head、后处理语义、checkpoint 字段或指标定义的变更必须包含：

1. 明确的兼容性说明；
2. fixture 和 round-trip 测试；
3. 有意设计的 schema version 或迁移决策；
4. 英文和简体中文契约文档同步更新；
5. 旧产物无法加载时的明确限制说明。

设备、worker 数量和数据路径等运行参数可在 checkpoint 恢复契约允许时改变。协议改变后的 benchmark 不应被描述为旧运行的延续。
