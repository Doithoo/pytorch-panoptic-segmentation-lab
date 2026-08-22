# 数据格式

每个样本由三个 stem 相同的文件组成：

- `images/<id>.png`：RGB 图片。
- `semantic/<id>.png`：像素类别 ID，`255` 表示忽略。
- `instance/<id>.png`：stuff 和忽略区域为 `0`，thing 实例使用正整数 ID。

运行 `prepare-data` 后会生成包含 `sample_id`、`image_path`、`semantic_path`、`instance_path` 四列的 CSV。路径可以是绝对路径，也可以相对于 manifest 目录。

schema 中的类别 ID 必须从 0 连续排列，并通过 `isthing` 标识实例类别或 stuff 类别。PQ 评估时，thing 类别按不同实例匹配，stuff 类别按每类一个区域匹配。
