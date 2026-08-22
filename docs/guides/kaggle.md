# Kaggle GPU Guide

Use a Kaggle Notebook with a free GPU accelerator. Add a dataset with this layout:

```text
/kaggle/input/your-panoptic-dataset/
  images/
  semantic/
  instance/
```

Run from the repository root:

```bash
pip install -e .
python scripts/kaggle_train.py --input /kaggle/input/your-panoptic-dataset
```

The runner refuses to start without CUDA and stores manifests, checkpoints, and metrics in `/kaggle/working`. For free-tier sessions, reduce `data.image_size`, `data.batch_size`, and `train.epochs` before increasing them. Download `/kaggle/working/artifacts` after the session ends.
