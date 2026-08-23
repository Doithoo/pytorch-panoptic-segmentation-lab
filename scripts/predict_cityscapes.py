"""Export checkpoint predictions in Cityscapes-compatible panoptic format."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from panoptic_segmenter.data.cityscapes import (
    cityscapes_panoptic_ids,
    cityscapes_schema,
    cityscapes_segments_info,
    write_cityscapes_panoptic_json,
    write_cityscapes_panoptic_png,
)
from panoptic_segmenter.inference import Predictor


def _resolve(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def export_predictions(checkpoint: str | Path, manifest: str | Path, output: str | Path, device: str) -> Path:
    manifest_path = Path(manifest).resolve()
    output_root = Path(output).resolve()
    raw_root = output_root / "raw"
    panoptic_root = output_root / "panoptic"
    output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"manifest contains no samples: {manifest_path}")
    predictor = Predictor.from_checkpoint(checkpoint, device)
    expected_names = tuple(item.name for item in cityscapes_schema().classes)
    if tuple(item.name for item in predictor.schema.classes) != expected_names:
        raise ValueError("checkpoint schema is not the official 19-class Cityscapes schema")
    annotations: list[dict[str, object]] = []
    for row in rows:
        image_path = _resolve(manifest_path, row["image_path"])
        paths = predictor.predict_path(image_path, raw_root)
        with Image.open(paths["semantic"]) as opened:
            semantic = np.asarray(opened, dtype=np.uint8).copy()
        with Image.open(paths["instance"]) as opened:
            instance = np.asarray(opened, dtype=np.uint16).copy()
        with Image.open(image_path) as opened:
            width, height = opened.size
        file_name = f"{row['sample_id']}.png"
        encoded_path = write_cityscapes_panoptic_png(
            cityscapes_panoptic_ids(semantic, instance),
            panoptic_root / file_name,
        )
        annotations.append(
            {
                "image_id": row.get("provider_sample_id") or row["sample_id"],
                "file_name": encoded_path.name,
                "image_file_name": image_path.name,
                "width": width,
                "height": height,
                "segments_info": cityscapes_segments_info(semantic, instance),
            }
        )
    return write_cityscapes_panoptic_json(output_root / "predictions.json", annotations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="artifacts/cityscapes-predictions")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    print(export_predictions(args.checkpoint, args.manifest, args.output, args.device))
