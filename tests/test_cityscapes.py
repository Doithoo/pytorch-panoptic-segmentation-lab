from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from panoptic_segmenter.data import (
    cityscapes_panoptic_ids,
    cityscapes_schema,
    cityscapes_segments_info,
    convert_cityscapes_dataset,
    convert_cityscapes_labels,
    discover_cityscapes_root,
    inspect_prepared_dataset,
    read_cityscapes_panoptic_png,
    write_cityscapes_panoptic_png,
)


def _cityscapes_fixture(root: Path, split: str, city: str, base: str, offset: int) -> None:
    image_dir = root / "leftImg8bit" / split / city
    label_dir = root / "gtFine" / split / city
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (6, 4), (10, 20, 30))
    image.save(image_dir / f"{base}_leftImg8bit.png")
    labels = np.array(
        [
            [7, 7, 24, 24, 11, 255],
            [7, 8, 24, 24, 11, 11],
            [21, 21, 26, 26, 23, 23],
            [21, 21, 26, 26, 23, 23],
        ],
        dtype=np.uint8,
    )
    instances = np.zeros_like(labels, dtype=np.uint16)
    instances[labels == 24] = 24001 + offset
    instances[labels == 26] = 26002 + offset
    Image.fromarray(labels).save(label_dir / f"{base}_gtFine_labelIds.png")
    Image.fromarray(instances).save(label_dir / f"{base}_gtFine_instanceIds.png")


def test_cityscapes_label_mapping_and_instance_reindexing() -> None:
    labels = np.array([[7, 24, 24, 11, 255]], dtype=np.int64)
    instances = np.array([[0, 24001, 24001, 0, 0]], dtype=np.int64)
    semantic, instance = convert_cityscapes_labels(labels, instances)
    assert semantic.tolist() == [[0, 11, 11, 2, 255]]
    assert instance.tolist() == [[0, 1, 1, 0, 0]]
    assert cityscapes_schema().num_classes == 19
    assert cityscapes_schema().thing_ids == (11, 12, 13, 14, 15, 16, 17, 18)


def test_cityscapes_rejects_mismatched_thing_category() -> None:
    labels = np.array([[24]], dtype=np.int64)
    instances = np.array([[26001]], dtype=np.int64)
    with pytest.raises(ValueError, match="expected 24"):
        convert_cityscapes_labels(labels, instances)


def test_cityscapes_group_instance_becomes_ignore() -> None:
    labels = np.array([[24, 24, 24]], dtype=np.int64)
    instances = np.array([[24, 24000, 24001]], dtype=np.int64)
    semantic, instance = convert_cityscapes_labels(labels, instances)
    assert semantic.tolist() == [[255, 255, 11]]
    assert instance.tolist() == [[0, 0, 1]]


def test_cityscapes_panoptic_rgb_roundtrip_and_segments(tmp_path: Path) -> None:
    semantic = np.array([[0, 11, 11], [2, 2, 255]], dtype=np.uint8)
    instance = np.array([[0, 1, 1], [0, 0, 0]], dtype=np.uint16)
    encoded = cityscapes_panoptic_ids(semantic, instance)
    path = write_cityscapes_panoptic_png(encoded, tmp_path / "sample.png")
    assert np.array_equal(read_cityscapes_panoptic_png(path), encoded)
    segments = cityscapes_segments_info(semantic, instance)
    assert {item["category_id"] for item in segments} == {7, 11, 24}
    assert sum(item["area"] for item in segments) == 5


def test_cityscapes_root_discovery_requires_one_official_tree(tmp_path: Path) -> None:
    root = tmp_path / "input" / "private-cityscapes"
    (root / "leftImg8bit").mkdir(parents=True)
    (root / "gtFine").mkdir()
    assert discover_cityscapes_root(tmp_path / "input") == root.resolve()
    second = tmp_path / "input" / "duplicate"
    (second / "leftImg8bit").mkdir(parents=True)
    (second / "gtFine").mkdir()
    with pytest.raises(ValueError, match="exactly one"):
        discover_cityscapes_root(tmp_path / "input")


def test_cityscapes_converter_preserves_official_splits_and_metadata(tmp_path: Path) -> None:
    source = tmp_path / "cityscapes"
    _cityscapes_fixture(source, "train", "aachen", "aachen_000000_000000", 0)
    _cityscapes_fixture(source, "val", "bochum", "bochum_000000_000000", 1)
    output = convert_cityscapes_dataset(source, tmp_path / "prepared")
    report = inspect_prepared_dataset(output)
    assert not report.issues
    assert report.split_counts == {"train": 1, "valid": 1, "test": 0}
    metadata = yaml.safe_load((output / "dataset.yaml").read_text(encoding="utf-8"))
    assert metadata["test_available"] is False
    assert metadata["official_splits"] == {"train": "train", "valid": "val", "test": None}
    assert (output / "panoptic_train.json").is_file()
    assert (output / "panoptic_valid.json").is_file()
    train_payload = json.loads((output / "panoptic_train.json").read_text(encoding="utf-8"))
    assert len(train_payload["categories"]) == 19
    assert train_payload["annotations"][0]["image_id"] == "aachen_000000_000000"
    with (output / "train.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["provider_sample_id"] == "aachen_000000_000000"
    assert row["source_split"] == "train"
