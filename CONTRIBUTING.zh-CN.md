# 贡献指南

[English](CONTRIBUTING.md)

贡献应保持学习路径可读，并让实验声明可审计。

1. 使用聚焦分支，行为变化必须增加测试。
2. 提交 PR 前运行 `make check`。
3. 不提交数据集、大 checkpoint、凭据、Kaggle token 或生成产物。
4. 新增 target、模型输出、指标、配置和 checkpoint 字段时，同时更新英文和简体中文文档。
5. 教程解释概念，指南给操作步骤，参考页定义精确契约，recorded run 只保存真实证据。

修改指标时加入可手算的 void、thing、stuff、缺席类别和多图聚合例子。修改模型时加入无需下载权重的 CPU shape/backward 测试。数据转换器必须包含错误标签 fixture、官方 split 证据、raw-to-train ID 表、crowd/void 行为和许可说明。

发布结果必须声明 resolved config、源码 revision、数据协议、checkpoint 哈希、环境、选择规则、耗时和限制。不得把占位或估算指标写成完成结果。

checkpoint 加载、模型重建、压缩包解压、子进程、依赖安装、Kaggle runner 和路径解析都属于安全敏感变更，PR 中应解释信任边界，变化时同步更新 `SECURITY.md`。
