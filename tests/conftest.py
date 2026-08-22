from __future__ import annotations

from pathlib import Path

import pytest

from panoptic_segmenter.config import ExperimentConfig, load_config
from panoptic_segmenter.data import default_label_schema
from panoptic_segmenter.data.manifest import prepare_paired_dataset
from panoptic_segmenter.data.synthetic import create_synthetic_dataset


@pytest.fixture
def prepared_experiment(tmp_path: Path) -> tuple[ExperimentConfig, Path]:
    data = tmp_path / "raw"
    manifests = tmp_path / "manifests"
    create_synthetic_dataset(data, count=8, size=32)
    prepare_paired_dataset(data, manifests, schema=default_label_schema())
    config = load_config(
        overrides={
            "data.data_dir": str(data),
            "data.manifest_dir": str(manifests),
            "data.image_size": [32, 32],
            "data.batch_size": 2,
            "data.center_sigma": 2.0,
            "data.max_train_samples": None,
            "data.max_valid_samples": None,
            "data.max_test_samples": None,
            "model.base_channels": 4,
            "train.epochs": 1,
            "train.amp": False,
            "train.scheduler": "none",
            "postprocess.instance_area": 1,
            "postprocess.stuff_area": 1,
            "output_dir": str(tmp_path / "artifacts"),
            "run_name": "test-run",
            "device": "cpu",
        }
    )
    return config, tmp_path
