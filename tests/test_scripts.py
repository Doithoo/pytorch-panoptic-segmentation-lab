from __future__ import annotations

from pathlib import Path

from PIL import Image

from panoptic_segmenter.data import default_label_schema
from panoptic_segmenter.data.manifest import prepare_paired_dataset
from panoptic_segmenter.data.preview import create_preview
from panoptic_segmenter.data.synthetic import create_synthetic_dataset


def test_preview_panoptic_creates_contact_sheet(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    manifests = tmp_path / "manifests"
    create_synthetic_dataset(raw, count=4, size=32)
    prepare_paired_dataset(raw, manifests, schema=default_label_schema())
    output = create_preview(manifests / "train.csv", tmp_path / "preview.png", limit=2)
    assert output.is_file()
    assert Image.open(output).size == (960, 536)
