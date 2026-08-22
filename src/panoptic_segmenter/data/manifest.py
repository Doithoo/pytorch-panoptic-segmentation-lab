"""Prepare manifests from a simple three-folder dataset."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from .schema import LabelSchema


def prepare_paired_dataset(
    data_dir: str | Path,
    manifest_dir: str | Path,
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    schema: LabelSchema,
) -> dict[str, Path]:
    root, output = Path(data_dir), Path(manifest_dir)
    image_dir, semantic_dir, instance_dir = root / "images", root / "semantic", root / "instance"
    images = {path.stem: path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    semantic = {path.stem: path for path in semantic_dir.iterdir() if path.suffix.lower() == ".png"}
    instance = {path.stem: path for path in instance_dir.iterdir() if path.suffix.lower() == ".png"}
    ids = sorted(set(images) & set(semantic) & set(instance))
    if not ids:
        raise ValueError("no matching image, semantic, and instance files found")
    random.Random(seed).shuffle(ids)
    first = int(len(ids) * ratios[0])
    second = first + int(len(ids) * ratios[1])
    splits = {"train": ids[:first], "valid": ids[first:second], "test": ids[second:]}
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for split, split_ids in splits.items():
        path = output / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_id", "image_path", "semantic_path", "instance_path"))
            writer.writeheader()
            for sample_id in split_ids:
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "image_path": str(images[sample_id].resolve()),
                        "semantic_path": str(semantic[sample_id].resolve()),
                        "instance_path": str(instance[sample_id].resolve()),
                    }
                )
        result[split] = path
    schema.write_yaml(output / "schema.yaml")
    return result
