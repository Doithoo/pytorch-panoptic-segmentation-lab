"""Prepared panoptic sample previews."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from ..evaluation.visualization import colorize_semantic, panoptic_overlay
from .schema import LabelSchema


def _resolve(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def create_preview(manifest: str | Path, output: str | Path, limit: int = 4) -> Path:
    """Write a contact sheet with source, semantic, and panoptic views."""
    if limit < 1:
        raise ValueError("limit must be positive")
    manifest_path = Path(manifest).resolve()
    schema = LabelSchema.read_yaml(manifest_path.parent / "schema.yaml")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:limit]
    if not rows:
        raise ValueError(f"manifest contains no samples: {manifest_path}")

    tile_width, tile_height = 320, 240
    labels = ("image", "semantic", "panoptic")
    sheet = Image.new("RGB", (tile_width * len(labels), (tile_height + 28) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        with Image.open(_resolve(manifest_path, row["image_path"])) as opened:
            image = opened.convert("RGB")
        with Image.open(_resolve(manifest_path, row["semantic_path"])) as opened:
            semantic = np.asarray(opened, dtype=np.uint8).copy()
        with Image.open(_resolve(manifest_path, row["instance_path"])) as opened:
            instance = np.asarray(opened, dtype=np.int64).copy()
        panels = (image, colorize_semantic(semantic, schema), panoptic_overlay(image, semantic, instance, schema))
        y = row_index * (tile_height + 28)
        draw.text((4, y + 4), row["sample_id"], fill="black")
        for column, (label, panel) in enumerate(zip(labels, panels, strict=True)):
            x = column * tile_width
            draw.text((x + 4, y + 4), label, fill="black")
            fitted = ImageOps.contain(panel, (tile_width - 8, tile_height - 8))
            sheet.paste(fitted, (x + (tile_width - fitted.width) // 2, y + 28 + (tile_height - fitted.height) // 2))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return destination
