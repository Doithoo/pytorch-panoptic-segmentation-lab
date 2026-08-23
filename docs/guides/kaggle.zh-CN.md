# 在 Kaggle 完成训练

[English](kaggle.md) | [参考运行状态](../recorded-run/README.zh-CN.md)

仓库提供的 kernel 无需附加 Kaggle Dataset，会完成固定 256 张合成图的参考运行。对于小型公开数据教学流程，请使用 [Kaggle Soccer 指南](kaggle-soccer.zh-CN.md)，它会在训练前转换 `quantigoai/soccer-dataset`。合成流程验证非交互 GPU 与产物流程；两个流程都不是官方 benchmark。

## 准备

单独安装 Kaggle CLI 并登录：

```bash
uv tool install kaggle
kaggle auth login
kaggle --version
```

提交前先把仓库改动 push 到 GitHub，因为 kernel 会 clone 仓库。打开 `docs/recorded-run/kaggle/kernel-metadata.json`，替换 `your-username`，保持 GPU 和 Internet 开启。

## 提交与监控

```bash
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status <username>/pytorch-panoptic-segmentation-lab-gpu
```

runner 默认 checkout `main`，记录实际 commit，安装 package、生成数据、执行 CUDA 预检、训练、重载 `best.pt`、评估 test 并写摘要。正式参考运行前，建议把 `run_kaggle.py` 的 `REVISION` 改为已 push 的 commit SHA。

请选择 T4 或更新 GPU，不要选 P100；当前 Kaggle PyTorch 可能不含其 `sm_60` kernel。runner 会真实执行一次 CUDA forward/backward，而不只检查 `torch.cuda.is_available()`。

健康日志依次出现 `source`、`preflight`、epoch、每 60 秒 `training` 心跳和 `complete`。心跳只表示进程存活，不表示 epoch 已完成。

## 下载与检查

```bash
kaggle kernels output <username>/pytorch-panoptic-segmentation-lab-gpu \
  --file-pattern 'artifacts/.*|kaggle-run-summary.json' -p kaggle-output
```

重点检查：

| 文件 | 检查内容 |
|---|---|
| `kaggle-run-summary.json` | complete、GPU、revision、耗时、split、test 指标、checkpoint 哈希 |
| `config.yaml` | 实际 CUDA/AMP/data/postprocess 配置 |
| `run.yaml` | 环境、数据身份、Git revision、时间 |
| `metrics.csv` | 20 行、有限 loss 分量和 validation 指标 |
| `best.pt` / `last.pt` | 验证集最优与最终可恢复状态 |
| `evaluation/evaluation.json` | 从 `best.pt` 自动生成的 test 摘要 |
| `evaluation/evaluation_detailed.json` | checkpoint/data identity、每图指标和最低 PQ 样本 |
| `evaluation/per_class.csv` | 每个 schema 类别的 PQ/SQ/RQ |

把小型证据回填到 `docs/recorded-run/` 并更新中英文 README 和 Kaggle 页面链接，不提交大型 checkpoint。

## 使用真实数据

若数据已经符合三目录契约，附加一个私有 Kaggle Dataset，并调用 `scripts/kaggle_train.py --input /kaggle/input/<dataset>`；同时用 `--schema` 传匹配 schema，并确保 config 类别数和 ignore 一致。

Cityscapes/COCO 原始格式不能直接使用。可信真实结果还需要：明确转换器与连续 ID 映射、官方 split、数据集特定 crowd/void 和官方 evaluator 对拍、数据身份与许可、per-class 和错误可视化。Cityscapes test 标签不公开，只能报告 validation 或提交官方服务器，不能把 validation 改称 test。

## 常见失败

- 仓库找不到：先 push 或修正 `REPOSITORY`。
- checkout 失败：固定远端已存在的 commit。
- 无 CUDA：启用 GPU 并重启 kernel。
- P100 kernel 失败：切换 T4 或更新 GPU。
- OOM：先减 batch，再减图像尺寸或模型宽度；center top-k 已限制解码显存。
- resume mismatch：保持数据/schema/config，只增加 epoch。
- stem 不匹配：训练前修复三个目录。

首次合成结果证明复现机制；完成额外 benchmark 协议后，才发布真实基准声明。
