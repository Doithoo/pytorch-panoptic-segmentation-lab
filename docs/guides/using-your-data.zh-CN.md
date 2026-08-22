# 使用自己的数据

[English](using-your-data.md) | [数据参考](../reference/data-format.zh-CN.md)

1. 选择连续训练 ID `0..C-1`、范围外 ignore ID 和稳定 thing/stuff 定义。
2. 转换为同 stem 的 `images`、`semantic`、`instance` 三目录。
3. 编写包含唯一名称、RGB 颜色和 `isthing` 的 schema。
4. 先 prepare/inspect，再改模型配置。

```bash
uv run panoptic-segment prepare-data --data-dir /path/to/raw \
  --manifest-dir data/my-manifests --schema configs/my_schema.yaml
uv run panoptic-segment inspect-data --manifest-dir data/my-manifests
```

随后设置 `data.manifest_dir`、`model.expected_num_classes`、`loss.ignore_index`。有官方 split 时必须保留；通用 preparer 只生成独立样本随机 split，不是 benchmark converter。

转换器应是独立、带测试的模块，记录 raw ID 映射、void/crowd、instance 编码、官方 split 来源和许可。ID 可能超过 255 时使用 16-bit PNG 或整数模式；不要通过颜色插值编码实例边界。

长训练前检查类别/实例频率和多张 overlay。结构预检通过也无法发现系统性错误映射或语义通道互换。
