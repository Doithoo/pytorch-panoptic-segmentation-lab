# 配置值如何确定

[English](configuration-flow.md) | [配置参考](../reference/config-reference.zh-CN.md)

配置按下面的顺序覆盖：

```text
dataclass 默认值 < YAML 文件 < 多个 --set 参数 < --device
```

例如，下面的命令先读取包内默认值，再应用 `learning_minimal.yaml`，接着修改 batch size，最后指定使用 CPU：

```bash
uv run panoptic-segment show-config \
  --config configs/learning_minimal.yaml \
  --set data.batch_size=4 \
  --device cpu
```

`show-config` 打印最终生效的值。字段名不存在或取值无效时，程序会在加载数据前报错。路径会转换为 `Path`，图像尺寸会转换为两个元素的 tuple，最后统一检查各部分设置。

训练开始时，这份最终配置会写入运行目录的 `config.yaml`，也会保存在每个 checkpoint 中。评估和预测直接从 checkpoint 读取图像尺寸、后处理阈值等设置。原始 YAML 表示你提交的配置，运行目录中的 `config.yaml` 才表示程序实际使用的配置。

部分设置还必须与准备好的数据一致：

- `model.expected_num_classes` 必须等于 `schema.yaml` 中的类别数；
- `loss.ignore_index` 必须等于 schema 中的忽略值；
- 断点恢复和评估时，当前 `dataset.yaml` 的数据指纹必须与 checkpoint 一致。

修改实验设置时使用新的 `run_name`。训练不会覆盖已有的指标文件；继续同一次运行时，通过 `--resume` 加载该目录下的 `last.pt`。
