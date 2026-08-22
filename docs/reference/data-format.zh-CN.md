# 数据格式与 Prepared Metadata

[English](data-format.md) | [使用自己的数据](../guides/using-your-data.zh-CN.md)

## 源目录

每个样本由同 stem 文件组成：

- `images/<id>.png|jpg|jpeg`：可解码为 RGB 的图像。
- `semantic/<id>.png`：二维 class ID，schema ignore 不参与训练。
- `instance/<id>.png`：二维整数 ID，stuff/ignore 为 0，thing 为正。

class ID 从 0 连续，名称唯一，每类包含 `isthing` 和 RGB 展示色。正 instance ID 只在图内有意义，并且只映射一个 thing 类。thing 不能为 0，stuff/ignore 不能为正。格式 v1 不表达 crowd。

ID 可能超过 255 时使用 16-bit PNG 或 Pillow 整数模式。mask 是标签，不要使用有损压缩或 bilinear 插值。

## Manifest CSV

`train.csv`、`valid.csv`、`test.csv` 包含：

```text
sample_id,image_path,semantic_path,instance_path
```

默认写相对 manifest 目录的路径，loader 也接受绝对路径。sample ID 不能跨 split 重复。

## `schema.yaml`

```yaml
classes:
  - id: 0
    name: road
    isthing: false
    color: [80, 120, 180]
  - id: 1
    name: person
    isthing: true
    color: [230, 80, 80]
ignore_index: 255
```

## `dataset.yaml`

格式 v1 记录 `data_dir`、seed、比例、split 数、manifest SHA-256、schema SHA-256 和 `identity`。identity 哈希 prepared manifest 与 schema；预检会解码源图，但不会逐字节哈希全部图片，发布 benchmark 时仍需记录外部数据版本。

`summary.txt` 只是便于阅读的 identity/split 摘要，不能代替 `dataset.yaml`。
