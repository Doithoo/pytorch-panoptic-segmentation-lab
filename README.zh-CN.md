# PyTorch Panoptic Segmentation Lab

一个面向学习与复现实验的全景分割开源项目。它把图像分割项目的“像素级 logits、可复现配置、可视化和训练记录”与目标检测项目的“实例级评估和推理工作流”连接起来，但核心实现独立维护。

## 当前实现

- Panoptic-DeepLab 风格轻量 U-Net：共享 encoder，输出 semantic logits、center heatmap、2-channel offset。
- 通用三件套数据格式：`image + semantic mask + instance mask`，不绑定特定数据集。
- 后处理：中心点 NMS、offset 最近中心分配、thing/stuff 规则。
- 指标：Panoptic Quality `PQ`、Segmentation Quality `SQ`、Recognition Quality `RQ`，以及训练损失。
- 工程入口：`show-config`、`prepare-data`、`train`、`evaluate`、`predict`。
- Kaggle：GPU 预检和 `/kaggle/input/<dataset>/{images,semantic,instance}` 数据布局。

## 快速开始

```bash
cd pytorch-panoptic-segmentation-lab
uv sync --extra dev
uv run python scripts/create_synthetic_data.py
uv run panoptic-segment prepare-data --schema configs/cityscapes_mini_schema.yaml
uv run panoptic-segment train --config configs/learning_minimal.yaml --dry-run
uv run panoptic-segment train --config configs/learning_minimal.yaml --device cpu
```

训练结果写入 `artifacts/<run_name>/`，包括 `config.yaml`、`metrics.csv`、`last.pt` 和 `best.pt`。

## 数据格式

```text
data/raw/
  images/sample_0001.png
  semantic/sample_0001.png   # 每个像素是 class id
  instance/sample_0001.png   # stuff 为 0，thing 实例为 1, 2, ...
```

三类文件的 stem 必须一致。`configs/cityscapes_mini_schema.yaml` 中的 `isthing` 决定一个类别是按实例评估还是整类 stuff 评估。真实数据可直接按相同结构整理；大于 255 的实例 ID 使用 16-bit PNG 或 PIL 的整数模式保存。

## Kaggle 免费 GPU

1. 将项目上传到 Kaggle Notebook，或者把仓库作为输入数据集添加。
2. 添加一个数据集，其根目录包含 `images/`、`semantic/`、`instance/`。
3. Notebook 中执行：

```bash
!pip install -e /kaggle/working/pytorch-panoptic-segmentation-lab
!python scripts/kaggle_train.py --input /kaggle/input/your-panoptic-dataset
```

在 Kaggle 右侧 `Settings -> Accelerator` 选择可用的免费 GPU。脚本会检查 CUDA、打印 GPU 名称、在 `/kaggle/working/manifests` 生成清单，并将 checkpoint 和 metrics 写入 `/kaggle/working/artifacts`。免费额度下建议从 `learning_minimal.yaml` 开始，再逐步增大 `image_size`、`batch_size` 和 `epochs`。

## 学习路径

1. 阅读 `docs/reference/data-format.zh-CN.md`，理解 semantic/instance 的目标契约。
2. 运行 `examples/01_panoptic_target.py`，观察中心点和 offset。
3. 阅读 `src/panoptic_segmenter/models/panoptic_unet.py` 与 `training/losses.py`。
4. 用 `evaluate` 对比 PQ、SQ、RQ，再用 `predict` 导出两张 mask。

项目目前以清晰可读的基线为主，后续可加入 DeepLabV3、可替换 backbone、COCO panoptic JSON 转换器和更完整的可视化。
