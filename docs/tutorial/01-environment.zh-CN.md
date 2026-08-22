# 01：环境与 CLI

[English](01-environment.md) | [上一节](00-basics.zh-CN.md) | [下一节](02-data-and-targets.zh-CN.md)

支持 Python 3.10–3.12。使用 `uv` 保持 lockfile 与环境一致：

```bash
uv sync --locked --extra dev
uv run panoptic-segment --version
make check
```

`make check` 会执行 lint、格式检查、mypy、测试和构建。learning-minimal 不要求 GPU，可用 `--device cpu`、`mps` 或 `cuda` 明确选择。

CLI 职责：

| 命令 | 职责 |
|---|---|
| `show-config` | 合并默认值、YAML 和 `--set` 后打印 |
| `prepare-data` | 配对源文件并生成确定性 manifest |
| `inspect-data` | 在训练前验证标签 |
| `train` | dry run、开始或恢复训练 |
| `evaluate` | 安全加载 checkpoint 并评估 split |
| `predict` | 保持原图尺寸，导出 mask 和可视化 |

wheel 安装后不依赖仓库相对 YAML，因此在任意目录均可执行 `show-config`。仓库示例仍显式传配置，方便审计实验。

小范围实验用 `--set key=value`。YAML 解析会保留 `null`、布尔、数字和列表类型；未知字段和非法值会在加载数据前失败。
