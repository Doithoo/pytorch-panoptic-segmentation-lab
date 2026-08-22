# 配置流

[English](configuration-flow.md) | [配置参考](../reference/config-reference.zh-CN.md)

解析优先级：

```text
dataclass 默认值 < YAML < 重复 --set < 显式 --device
```

未知 YAML/override 字段会失败。路径转为 `Path`，image size 转为 tuple，重建全部 dataclass 后统一校验。`show-config` 打印的就是该 resolved object。

resolved config 会写入运行目录并嵌入每个 checkpoint；评估和预测从 checkpoint 恢复它，包括后处理阈值。源 YAML 是输入，运行目录中的 `config.yaml` 才是本次实验的权威记录。

部分事实有外部一致性检查：model 类别数匹配 schema，loss ignore 匹配 schema，恢复时 dataset identity 匹配 `dataset.yaml`。这能防止语法合法的配置改变标签含义。

实验应使用不同 run name。普通训练拒绝覆盖已有 metrics，继续训练必须显式传 `--resume`。
