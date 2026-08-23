from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from panoptic_segmenter.data import default_label_schema, inspect_prepared_dataset
from panoptic_segmenter.data.manifest import prepare_paired_dataset
from panoptic_segmenter.data.synthetic import create_synthetic_dataset


def test_prepare_data_creates_nonempty_portable_splits(tmp_path: Path) -> None:
    raw, manifests = tmp_path / "raw", tmp_path / "manifests"
    create_synthetic_dataset(raw, count=5, size=32)
    paths = prepare_paired_dataset(raw, manifests, schema=default_label_schema())
    counts: dict[str, int] = {}
    for split, path in paths.items():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        counts[split] = len(rows)
        assert rows
        assert not Path(rows[0]["image_path"]).is_absolute()
    assert counts == {"train": 3, "valid": 1, "test": 1}
    assert (manifests / "dataset.yaml").is_file()
    assert not inspect_prepared_dataset(manifests).issues


def test_prepare_data_keeps_groups_together(tmp_path: Path) -> None:
    raw, manifests = tmp_path / "raw", tmp_path / "manifests"
    create_synthetic_dataset(raw, count=6, size=32)
    with (raw / "groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "group_id"))
        writer.writeheader()
        for index in range(6):
            writer.writerow({"sample_id": f"sample_{index:04d}", "group_id": f"video-{index // 2}"})
    prepare_paired_dataset(raw, manifests, schema=default_label_schema(), group_file=raw / "groups.csv")
    assert not inspect_prepared_dataset(manifests).issues
    split_groups: dict[str, set[str]] = {}
    for split in ("train", "valid", "test"):
        with (manifests / f"{split}.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        split_groups[split] = {f"video-{int(row['sample_id'].split('_')[-1]) // 2}" for row in rows}
    assert sum(len(groups) for groups in split_groups.values()) == 3
    assert not (split_groups["train"] & split_groups["valid"])
    assert not (split_groups["train"] & split_groups["test"])
    assert not (split_groups["valid"] & split_groups["test"])


def test_prepare_data_rejects_unmatched_stems(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    create_synthetic_dataset(raw, count=3, size=32)
    (raw / "semantic/sample_0002.png").unlink()
    with pytest.raises(ValueError, match="stems do not match"):
        prepare_paired_dataset(raw, tmp_path / "manifests", schema=default_label_schema())


def test_prepare_data_rejects_invalid_ratios(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    create_synthetic_dataset(raw, count=3, size=32)
    with pytest.raises(ValueError, match="sum to one"):
        prepare_paired_dataset(raw, tmp_path / "manifests", schema=default_label_schema(), ratios=(0.8, 0.8, 0.1))


def test_inspection_detects_thing_pixels_without_instance_id(tmp_path: Path) -> None:
    raw, manifests = tmp_path / "raw", tmp_path / "manifests"
    create_synthetic_dataset(raw, count=3, size=32)
    prepare_paired_dataset(raw, manifests, schema=default_label_schema())
    with (manifests / "train.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    semantic_path = (manifests / row["semantic_path"]).resolve()
    instance_path = (manifests / row["instance_path"]).resolve()
    semantic = np.asarray(Image.open(semantic_path))
    instance = np.asarray(Image.open(instance_path)).copy()
    instance[semantic == 1] = 0
    Image.fromarray(instance.astype(np.uint16)).save(instance_path)
    report = inspect_prepared_dataset(manifests)
    assert any("thing pixels" in issue.message for issue in report.issues)


def test_synthetic_generator_supports_documented_minimum_size(tmp_path: Path) -> None:
    create_synthetic_dataset(tmp_path / "raw", count=3, size=32)
    assert Image.open(tmp_path / "raw/images/sample_0000.png").size == (32, 32)
