# 添加数据集转换器

[English](adding-datasets.md) | [数据格式](../reference/data-format.zh-CN.md) | [使用自己的数据](using-your-data.zh-CN.md)

只有在样本相互独立且允许随机切分时，才使用通用 `prepare-data`。benchmark 或从视频派生的数据集应有独立转换器，保留官方 split 和评测协议。

## 转换器契约

转换器应生成标准目录或等价 manifest 行：

```text
images/<sample>.png
semantic/<sample>.png
instance/<sample>.png
train.csv
valid.csv
test.csv
schema.yaml
dataset.yaml
```

必须记录并测试：

- raw ID 到连续 class ID 的映射；
- thing/stuff/crowd/void 行为；
- instance 编码和重新编号；
- 官方 split 成员和数据源版本；
- 图像与 mask 的尺寸、dtype 和插值规则；
- 许可与再分发边界；
- 官方 evaluator 需要时的 panoptic JSON/PNG 导出规则。

## 步骤

1. 将原始数据放在 Git 之外，并明确 source root。
2. 在处理完整数据前先转换一个人工检查样本。
3. 增加错误标签 fixture：未知 ID、尺寸不匹配、instance 类别不一致、group/crowd 区域和非正 thing ID。
4. 如果官方 evaluator 需要，保留 provider sample ID 到 manifest 列。
5. 如果转换器会被复用，使用 `register_converter()` 注册 provider。
6. 运行 `inspect-data`，检查 semantic、instance 和 overlay 预览。
7. 每次运行记录数据版本、split 政策、许可和转换器 revision。

不要对已经转换的 benchmark 目录再次调用 `prepare-data`，否则会随机化官方 split。没有通过数据集自身的评测政策验证导出文件时，不要把内部 PQ 结果称为官方分数。
