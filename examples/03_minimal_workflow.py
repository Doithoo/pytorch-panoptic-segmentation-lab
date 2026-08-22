"""Run the smallest complete local workflow."""

from pathlib import Path

from scripts.create_synthetic_data import create_dataset

from panoptic_segmenter.config import load_config
from panoptic_segmenter.data import LabelSchema
from panoptic_segmenter.data.manifest import prepare_paired_dataset
from panoptic_segmenter.training.train import train_from_config

root = Path("data/raw")
create_dataset(root, count=12)
schema = LabelSchema.read_yaml("configs/cityscapes_mini_schema.yaml")
prepare_paired_dataset(root, "data/manifests", schema=schema)
config = load_config("configs/learning_minimal.yaml")
print(train_from_config(config, dry_run=True))
