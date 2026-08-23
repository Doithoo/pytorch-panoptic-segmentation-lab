# Cityscapes Kaggle Run Status

[简体中文](README.zh-CN.md) | [Cityscapes guide](../../guides/cityscapes.md)

## Status: waiting for a licensed private dataset attachment

The conversion, training, prediction export, official ground-truth generation, and `cityscapesscripts` evaluation runner are implemented. No real Cityscapes metric is recorded here because this repository and the authenticated Kaggle account do not contain an official licensed Cityscapes archive with both `leftImg8bit` and `gtFine instanceIds`.

Public Kaggle search results commonly contain pix2pix image pairs or semantic masks rather than the complete official panoptic source tree. The project will not use an unverified re-upload to manufacture a benchmark result.

To complete the run:

1. Download `leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` from the official Cityscapes portal under its license.
2. Upload them, or their extracted tree, as a private Kaggle Dataset that is not redistributed.
3. Attach that Dataset to the private kernel described by [`kaggle/kernel-metadata.json`](kaggle/kernel-metadata.json).
4. Pin [`kaggle/run_kaggle.py`](kaggle/run_kaggle.py) to the reviewed repository commit.
5. Submit the kernel and retain `cityscapes-run-summary.json`, resolved config, metrics, official validation JSON, checkpoint hash, and selected visual errors.

The runner auto-discovers exactly one root containing `leftImg8bit/` and `gtFine/`, preserves official train/val, trains on train, selects on val, exports full-resolution masks from saved-size inference, generates crowd-aware official ground truth, and evaluates through `cityscapesscripts`. It never labels val as test.
