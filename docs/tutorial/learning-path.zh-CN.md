# 全景分割学习路线

[English](learning-path.md) | [文档目录](../README.zh-CN.md)

如果你已经了解基础 Tensor、卷积和梯度下降，这条路线大约需要 8-12 小时。先用本地生成的图像理解数据和模型，再进入公开视频数据流程。任何一步都可以停下来阅读源码。

## 1. 先让环境运行起来

```bash
uv sync --locked --extra dev
uv run panoptic-segment --version
uv run panoptic-segment show-config --config configs/learning_minimal.yaml
make check
```

打开打印出来的配置，找到输入尺寸、样本上限、各项损失权重、后处理阈值、设备，以及选择 `best.pt` 使用的指标。

## 2. 先看标签，再看模型

```bash
uv run python scripts/create_synthetic_data.py --count 24 --size 128
uv run panoptic-segment prepare-data --schema configs/synthetic_schema.yaml
uv run panoptic-segment inspect-data
uv run python scripts/preview_panoptic.py data/manifests/train.csv \
  --output artifacts/learning-preview.png --limit 4
uv run python examples/01_panoptic_target.py
```

回答下面几个问题：

- `semantic[y, x]` 保存什么？
- 为什么 thing 像素必须有正整数 instance ID？
- 为什么 instance mask 不作为模型输入？
- center heatmap 和两个 offset 通道分别指向哪里？

确认预览图的颜色、对象边界和 instance ID 都符合预期后，再继续下一步。

## 3. 执行一次参数更新

```bash
uv run python examples/02_model_contract.py
uv run panoptic-segment train --config configs/learning_minimal.yaml --dry-run
```

先看模型输出 shape，再沿着 CLI、模型、损失、反向传播和 optimizer step 读一遍调用过程。dry-run 只能说明各模块接通了，不能说明模型已经学会了分割。

## 4. 训练并查看产物

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml
```

按下面顺序打开文件：

1. `config.yaml`：这次实际使用了哪些设置；
2. `run.yaml`：记录了软件、设备、revision 和数据指纹；
3. `metrics.csv`：损失和验证指标如何变化；
4. `best.pt`：验证集选出的 checkpoint；
5. `last.pt`：用于继续训练的最新 checkpoint。

解释为什么 `best.pt` 和 `last.pt` 可能不同，以及为什么 test 不应该参与模型选择。

## 5. 把数字和像素放在一起看

```bash
uv run panoptic-segment evaluate artifacts/learning-minimal/best.pt --split test \
  --output artifacts/learning-minimal/evaluation.json
uv run panoptic-segment predict artifacts/learning-minimal/best.pt \
  data/raw/images/sample_0000.png --output artifacts/prediction
```

对比 `pq`、`pq_thing` 和 `pq_stuff`，再打开 semantic-color 和 overlay。PQ 较低时，分别检查类别错误、中心漏检或重复、offset 偏差、实例合并、实例拆分和面积过滤。

## 6. 修改一个设置并继续训练

```bash
uv run panoptic-segment train --config configs/learning_minimal.yaml \
  --set train.epochs=4 --resume artifacts/learning-minimal/last.pt
```

checkpoint 会恢复模型、optimizer、scheduler、随机数状态、指标历史和之前的 best 值。如果新的标签、target、后处理或数据划分不同，程序会拒绝恢复。

## 7. 使用真实的公开数据

按照 [Kaggle Soccer](../guides/kaggle-soccer.zh-CN.md) 操作。转换器从视频和 COCO 多边形开始，抽取帧、生成 semantic 和 instance mask，按源视频划分数据，然后继续使用相同的训练命令。

已记录运行的 test PQ 是 `0.223444`，thing PQ 是 `0.000000`。这个结果很适合用来观察从头训练的小模型哪里失败。先查看按类别指标和 worst-case 报告，再决定是否修改模型。

## 8. 扩展项目

当你能从原始标注一路跟踪到预测图后，选择一个小改动：

- 在 schema 中增加一个类别；
- 修改数据转换器；
- 注册另一个模型；
- 调整一个后处理参数；
- 在固定数据划分和 seed 下比较两次运行。

需要准确字段名或兼容性规则时，再查指南和参考资料。
