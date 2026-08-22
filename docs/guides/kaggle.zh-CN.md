# Kaggle GPU 使用

在 Kaggle Notebook 中启用免费 GPU，并添加如下目录结构的数据集：

```text
/kaggle/input/your-panoptic-dataset/
  images/
  semantic/
  instance/
```

在项目根目录执行：

```bash
pip install -e .
python scripts/kaggle_train.py --input /kaggle/input/your-panoptic-dataset
```

脚本会拒绝无 CUDA 的运行环境，并把 manifest、checkpoint、metrics 保存到 `/kaggle/working`。免费会话建议先降低 `data.image_size`、`data.batch_size` 和 `train.epochs`，训练结束后下载 `/kaggle/working/artifacts`。
