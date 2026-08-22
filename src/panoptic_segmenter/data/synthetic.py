"""Deterministic synthetic panoptic data used by tests and tutorials."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create_synthetic_dataset(root: str | Path, count: int = 24, size: int = 128) -> None:
    destination = Path(root)
    if count < 3:
        raise ValueError("count must be at least 3 so train, valid, and test are non-empty")
    if size < 32:
        raise ValueError("size must be at least 32 pixels")
    for name in ("images", "semantic", "instance"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    road_y = round(size * 0.69)
    first_width, first_height = round(size * 0.22), round(size * 0.30)
    second_width, second_height = round(size * 0.19), round(size * 0.25)
    for index in range(count):
        image = Image.new("RGB", (size, size), (70, 110, 165))
        semantic = Image.new("L", (size, size), 2)
        instance = Image.new("I", (size, size), 0)
        image_draw, semantic_draw, instance_draw = (
            ImageDraw.Draw(image),
            ImageDraw.Draw(semantic),
            ImageDraw.Draw(instance),
        )
        x = round(size * (0.09 + ((index * 7) % 48) / 128))
        y = round(size * (0.19 + ((index * 5) % 40) / 128))
        image_draw.rectangle((0, road_y, size, size), fill=(80, 80, 80))
        semantic_draw.rectangle((0, road_y, size, size), fill=0)
        image_draw.ellipse((x, y, x + first_width, y + first_height), fill=(220, 90, 80))
        semantic_draw.ellipse((x, y, x + first_width, y + first_height), fill=1)
        instance_draw.ellipse((x, y, x + first_width, y + first_height), fill=1)
        x2 = round(size * (0.55 - ((index * 3) % 22) / 128))
        y2 = round(size * 0.27)
        image_draw.ellipse((x2, y2, x2 + second_width, y2 + second_height), fill=(230, 170, 70))
        semantic_draw.ellipse((x2, y2, x2 + second_width, y2 + second_height), fill=1)
        instance_draw.ellipse((x2, y2, x2 + second_width, y2 + second_height), fill=2)
        image.save(destination / "images" / f"sample_{index:04d}.png")
        semantic.save(destination / "semantic" / f"sample_{index:04d}.png")
        Image.fromarray(np.asarray(instance, dtype=np.uint16)).save(
            destination / "instance" / f"sample_{index:04d}.png"
        )
