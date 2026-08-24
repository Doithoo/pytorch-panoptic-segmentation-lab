# 故障排查

[English](troubleshooting.md)

| 现象 | 检查什么 |
|---|---|
| `dataset stems do not match exactly` | 对比 `images`、`semantic` 和 `instance` 中的文件名。准备阶段不会静默丢弃文件。 |
| `split is empty` | 增加样本或调整比例。按组切分时，还要确认 group 数量足够。 |
| 一个 group 出现在多个 split | 检查 `groups.csv`；同一视频、场景、病人或源图像只能使用一个 group ID。 |
| thing/stuff 检查失败 | 修复转换器。thing 像素需要正 instance ID，stuff 和 void 像素需要 0。 |
| 图像尺寸不能被 16 整除 | 把训练配置中的 `data.image_size` 改为 128、256 或 `[256,512]` 等尺寸。源图像可以是其他尺寸。 |
| 类别数或 ignore 不一致 | 对照 `schema.yaml` 检查配置。 |
| 运行目录已存在 | 换一个 `run_name`，或从该目录的 `last.pt` 恢复。 |
| 恢复或评估时数据指纹不一致 | 使用原来的 manifest 和 schema；数据改变时开始新的运行。 |
| checkpoint 无法加载 | 文件可能损坏，或来自其他 schema version。不可信文件不要切换到 `weights_only=False`。 |
| CUDA 不可用 | 选择加速设备并重启进程。 |
| P100 CUDA kernel 报错 | 换用 T4 或更新的 GPU。 |
| 显存不足 | 降低 batch size、训练图像尺寸或 `base_channels`。 |
| semantic 看起来合理但 PQ 为 0 | 检查 center 峰值、阈值、offset、thing 标记和面积阈值。 |
| 预测结果变成 void | 没有同类别中心通过筛选，或分配到的区域被面积阈值过滤。 |
| wheel 命令找不到数据 | wheel 可以在任意目录打印默认配置，但训练仍需要在配置路径中找到 prepared manifest。 |

长时间运行前，先执行 `inspect-data`、打开预览图，再运行 `train --dry-run`。报告问题时附上命令、最终配置、traceback、Python/PyTorch 版本、设备和一个非私密样本。
