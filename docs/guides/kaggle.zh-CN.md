# 在 Kaggle 上运行 GPU 任务

[English](kaggle.md) | [Kaggle Soccer](kaggle-soccer.zh-CN.md) | [运行记录](../recorded-run/README.zh-CN.md)

仓库提供两条 Kaggle 路线：

- 下面的合成数据路线，用来检查安装包、CUDA、训练循环、checkpoint 重载和产物导出是否能连起来；
- [Kaggle Soccer](kaggle-soccer.zh-CN.md)，从公开视频数据开始，在训练前增加标注转换步骤。

两条路线都不是官方 Cityscapes 或 COCO benchmark。

## 提交前

安装并登录 Kaggle CLI：

```bash
uv tool install kaggle
kaggle auth login
kaggle --version
```

先把 kernel 要运行的源码 revision push 到 GitHub。打开 `docs/recorded-run/kaggle/kernel-metadata.json`，填入 Kaggle 用户名，并保持 GPU 和 Internet 开启。

## 提交和查看状态

```bash
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <username>/pytorch-panoptic-segmentation-lab-gpu
```

kernel 会 clone 仓库、切换到固定 revision、安装包、生成数据、检查 CUDA、训练、重载 `best.pt`、评估 test 并写出 summary。要保存一次可复查的结果，请固定完整 commit。

请选择 T4 或更新的 GPU。当前 Kaggle PyTorch 镜像不建议使用 P100，因为 CUDA build 可能没有 `sm_60` 所需的 kernel。runner 会执行真实的 CUDA forward 和 backward。

日志包含源码 revision、preflight、epoch、定时心跳和完成信息。心跳只表示进程还在运行，不代表某个 epoch 已经结束。

## 下载产物

状态变为 `COMPLETE` 后：

```bash
kaggle kernels output <username>/pytorch-panoptic-segmentation-lab-gpu \
  --file-pattern 'artifacts/.*|kaggle-run-summary.json' -p kaggle-output
```

检查：

| 文件 | 看什么 |
|---|---|
| `kaggle-run-summary.json` | 设备、源码 revision、耗时、划分、test 指标和 checkpoint hash |
| `config.yaml` | 最终的数据、模型、loss 和后处理设置 |
| `run.yaml` | 软件版本、数据指纹、Git revision 和耗时 |
| `metrics.csv` | 每轮训练记录和验证指标 |
| `best.pt` / `last.pt` | 选中的 checkpoint 和最新可恢复 checkpoint |
| `evaluation/evaluation.json` | test 总体指标 |
| `evaluation/evaluation_detailed.json` | 每图指标和最低 PQ 样本 |
| `evaluation/per_class.csv` | 每类 PQ/SQ/RQ |

只把小型结果文件回填到 `docs/recorded-run/`，checkpoint 和数据集保留在 Kaggle output 中。

## 换成其他数据

如果数据已经符合 `images/`、`semantic/`、`instance/` 三目录格式，可以挂载后运行：

```bash
python scripts/kaggle_train.py \
  --input /kaggle/input/<dataset> \
  --schema configs/my_schema.yaml \
  --config configs/my_config.yaml
```

如果原始数据是视频、COCO JSON、Cityscapes ID 或其他格式，先编写转换器。数据许可、类别映射、划分规则和 evaluator 应该与训练代码分开记录。

## 常见失败

- 仓库找不到：先 push 仓库或修正 `REPOSITORY`。
- checkout 失败：固定 GitHub 上存在的 commit。
- CUDA 不可用：启用 GPU 并重启 kernel。
- P100 kernel 失败：切换 T4 或更新的 GPU。
- 显存不足：降低 batch size、图像尺寸或 `base_channels`。
- resume 不匹配：保持数据、schema 和设置不变，只增加 epoch。
- stem 不匹配：训练前修复三个输入目录。

需要真实数据转换步骤时，使用 [Kaggle Soccer 指南](kaggle-soccer.zh-CN.md)。
