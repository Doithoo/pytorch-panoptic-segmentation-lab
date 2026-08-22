"""Run the reference training job on a Kaggle GPU kernel.

Expected Kaggle dataset input layout: /kaggle/input/<dataset>/{images,semantic,instance}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from panoptic_segmenter.config import load_config
from panoptic_segmenter.data import LabelSchema
from panoptic_segmenter.data.manifest import prepare_paired_dataset
from panoptic_segmenter.training.train import train_from_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="configs/learning_minimal.yaml")
    parser.add_argument("--output", default="/kaggle/working/artifacts")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; enable a T4/P100 GPU in Kaggle settings")
    input_root = Path(args.input)
    config = load_config(args.config)
    config.data.data_dir = input_root
    config.data.manifest_dir = Path("/kaggle/working/manifests")
    config.output_dir = Path(args.output)
    config.device = "cuda"
    schema = LabelSchema.read_yaml("configs/cityscapes_mini_schema.yaml")
    prepare_paired_dataset(input_root, config.data.manifest_dir, schema=schema)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"checkpoint directory: {train_from_config(config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
