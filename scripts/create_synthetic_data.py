"""Create a tiny synthetic panoptic dataset for local smoke tests and tutorials."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create_dataset(root: Path, count: int = 24, size: int = 128) -> None:
    for name in ("images", "semantic", "instance"):
        (root / name).mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = Image.new("RGB", (size, size), (70, 110, 165))
        semantic = Image.new("L", (size, size), 2)
        instance = Image.new("I", (size, size), 0)
        image_draw, semantic_draw, instance_draw = (
            ImageDraw.Draw(image),
            ImageDraw.Draw(semantic),
            ImageDraw.Draw(instance),
        )
        x = 12 + (index * 7) % 48
        y = 24 + (index * 5) % 40
        image_draw.rectangle((0, 88, size, size), fill=(80, 80, 80))
        semantic_draw.rectangle((0, 88, size, size), fill=0)
        image_draw.ellipse((x, y, x + 28, y + 38), fill=(220, 90, 80))
        semantic_draw.ellipse((x, y, x + 28, y + 38), fill=1)
        instance_draw.ellipse((x, y, x + 28, y + 38), fill=1)
        x2 = 70 - (index * 3) % 22
        image_draw.ellipse((x2, 34, x2 + 24, 66), fill=(230, 170, 70))
        semantic_draw.ellipse((x2, 34, x2 + 24, 66), fill=1)
        instance_draw.ellipse((x2, 34, x2 + 24, 66), fill=2)
        image.save(root / "images" / f"sample_{index:04d}.png")
        semantic.save(root / "semantic" / f"sample_{index:04d}.png")
        Image.fromarray(np.asarray(instance, dtype=np.uint16)).save(root / "instance" / f"sample_{index:04d}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()
    create_dataset(Path(args.output), args.count)
    print(f"created {args.count} samples in {args.output}")
