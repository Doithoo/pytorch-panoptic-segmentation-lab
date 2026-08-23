"""Cityscapes label conversion and official-split preparation.

This module accepts an extracted Cityscapes tree with ``leftImg8bit`` and
``gtFine`` directories. It writes the repository's three-mask contract while
preserving the official train/val membership.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from .schema import ClassDefinition, LabelSchema

CITYSCAPES_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CityscapesClass:
    raw_id: int
    name: str
    train_id: int
    has_instances: bool
    color: tuple[int, int, int]


# Official Cityscapes labelIds -> trainIds contract. Entries with train_id 255
# are ignored by the 19-class benchmark and are never instance targets.
CITYSCAPES_CLASSES: tuple[CityscapesClass, ...] = (
    CityscapesClass(0, "unlabeled", 255, False, (0, 0, 0)),
    CityscapesClass(1, "ego vehicle", 255, False, (0, 0, 0)),
    CityscapesClass(2, "rectification border", 255, False, (0, 0, 0)),
    CityscapesClass(3, "out of roi", 255, False, (0, 0, 0)),
    CityscapesClass(4, "static", 255, False, (0, 0, 0)),
    CityscapesClass(5, "dynamic", 255, False, (111, 74, 0)),
    CityscapesClass(6, "ground", 255, False, (81, 0, 81)),
    CityscapesClass(7, "road", 0, False, (128, 64, 128)),
    CityscapesClass(8, "sidewalk", 1, False, (244, 35, 232)),
    CityscapesClass(9, "parking", 255, False, (250, 170, 160)),
    CityscapesClass(10, "rail track", 255, False, (230, 150, 140)),
    CityscapesClass(11, "building", 2, False, (70, 70, 70)),
    CityscapesClass(12, "wall", 3, False, (102, 102, 156)),
    CityscapesClass(13, "fence", 4, False, (190, 153, 153)),
    CityscapesClass(14, "guard rail", 255, False, (180, 165, 180)),
    CityscapesClass(15, "bridge", 255, False, (150, 100, 100)),
    CityscapesClass(16, "tunnel", 255, False, (150, 120, 90)),
    CityscapesClass(17, "pole", 5, False, (153, 153, 153)),
    CityscapesClass(18, "polegroup", 255, False, (153, 153, 153)),
    CityscapesClass(19, "traffic light", 6, False, (250, 170, 30)),
    CityscapesClass(20, "traffic sign", 7, False, (220, 220, 0)),
    CityscapesClass(21, "vegetation", 8, False, (107, 142, 35)),
    CityscapesClass(22, "terrain", 9, False, (152, 251, 152)),
    CityscapesClass(23, "sky", 10, False, (70, 130, 180)),
    CityscapesClass(24, "person", 11, True, (220, 20, 60)),
    CityscapesClass(25, "rider", 12, True, (255, 0, 0)),
    CityscapesClass(26, "car", 13, True, (0, 0, 142)),
    CityscapesClass(27, "truck", 14, True, (0, 0, 70)),
    CityscapesClass(28, "bus", 15, True, (0, 60, 100)),
    CityscapesClass(29, "caravan", 255, True, (0, 0, 90)),
    CityscapesClass(30, "trailer", 255, True, (0, 0, 110)),
    CityscapesClass(31, "train", 16, True, (0, 80, 100)),
    CityscapesClass(32, "motorcycle", 17, True, (0, 0, 230)),
    CityscapesClass(33, "bicycle", 18, True, (119, 11, 32)),
)


def discover_cityscapes_root(input_root: str | Path) -> Path:
    """Find exactly one extracted Cityscapes tree below an input directory."""
    root = Path(input_root).resolve()
    candidates = [
        path
        for path in (root, *root.rglob("*"))
        if path.is_dir() and (path / "leftImg8bit").is_dir() and (path / "gtFine").is_dir()
    ]
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise ValueError(f"expected exactly one Cityscapes root below {root}, found {unique}")
    return unique[0]


def cityscapes_schema() -> LabelSchema:
    """Return the official 19-class Cityscapes train-ID schema."""
    valid = [item for item in CITYSCAPES_CLASSES if item.train_id != 255]
    return LabelSchema(
        classes=tuple(ClassDefinition(item.train_id, item.name, item.has_instances, item.color) for item in valid),
        ignore_index=255,
    )


def cityscapes_panoptic_ids(semantic: np.ndarray, instance: np.ndarray) -> np.ndarray:
    """Encode project train IDs as Cityscapes-style category*1000+instance IDs."""
    if semantic.shape != instance.shape or semantic.ndim != 2:
        raise ValueError("semantic and instance masks must be matching 2D arrays")
    train_to_raw = {item.train_id: item.raw_id for item in CITYSCAPES_CLASSES if item.train_id != 255}
    allowed = set(train_to_raw) | {255}
    unexpected = sorted({int(value) for value in np.unique(semantic)} - allowed)
    if unexpected:
        raise ValueError(f"semantic mask contains non-Cityscapes train IDs: {unexpected}")
    if np.any(instance < 0) or np.any(instance[semantic == 255] != 0):
        raise ValueError("instance mask is invalid on negative or void pixels")
    result = np.zeros(semantic.shape, dtype=np.uint32)
    for train_id, raw_id in train_to_raw.items():
        selected = semantic == train_id
        if not bool(selected.any()):
            continue
        if CITYSCAPES_CLASSES[raw_id].has_instances:
            if np.any(selected & (instance <= 0)):
                raise ValueError(f"thing train ID {train_id} contains non-positive instance IDs")
            result[selected] = raw_id * 1000 + instance[selected].astype(np.uint32)
        else:
            if np.any(selected & (instance != 0)):
                raise ValueError(f"stuff train ID {train_id} contains instance IDs")
            result[selected] = raw_id * 1000
    return result


def cityscapes_segments_info(semantic: np.ndarray, instance: np.ndarray) -> list[dict[str, Any]]:
    """Return COCO-panoptic-compatible segment metadata for one prediction."""
    encoded = cityscapes_panoptic_ids(semantic, instance)
    segments: list[dict[str, Any]] = []
    for segment_id in np.unique(encoded):
        segment_id_int = int(segment_id)
        if segment_id_int == 0:
            continue
        category_id = segment_id_int // 1000
        segments.append(
            {
                "id": segment_id_int,
                "category_id": category_id,
                "area": int((encoded == segment_id).sum()),
                "iscrowd": 0,
            }
        )
    return segments


def write_cityscapes_panoptic_png(encoded: np.ndarray, path: str | Path) -> Path:
    """Write uint32 panoptic IDs using the standard RGB encoding."""
    if encoded.ndim != 2 or encoded.dtype.kind not in "ui":
        raise ValueError("encoded panoptic IDs must be a two-dimensional integer array")
    values = encoded.astype(np.uint32)
    rgb = np.stack((values % 256, (values // 256) % 256, (values // 65536) % 256), axis=-1).astype(np.uint8)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(destination)
    return destination


def read_cityscapes_panoptic_png(path: str | Path) -> np.ndarray:
    with Image.open(path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint32)
    return rgb[:, :, 0] + 256 * rgb[:, :, 1] + 65536 * rgb[:, :, 2]


def cityscapes_categories() -> list[dict[str, Any]]:
    return [
        {"id": item.raw_id, "name": item.name, "supercategory": "cityscapes", "isthing": int(item.has_instances)}
        for item in CITYSCAPES_CLASSES
        if item.train_id != 255
    ]


def write_cityscapes_panoptic_json(path: str | Path, annotations: list[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for item in annotations:
        image: dict[str, Any] = {
            "id": item["image_id"],
            "file_name": item.get("image_file_name", item["file_name"]),
        }
        for field in ("width", "height"):
            if field in item:
                image[field] = item[field]
        images.append(image)
    clean_annotations = [
        {key: value for key, value in item.items() if key not in {"image_file_name", "width", "height"}}
        for item in annotations
    ]
    payload = {"images": images, "annotations": clean_annotations, "categories": cityscapes_categories()}
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def convert_cityscapes_labels(
    label_ids: np.ndarray,
    instance_ids: np.ndarray,
    *,
    strict: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one Cityscapes labelIds/instanceIds pair to train masks.

    Instance IDs are re-indexed from one per image. This avoids leaking the
    raw ``category_id * 1000 + instance_id`` encoding into the generic loader.
    """
    if label_ids.shape != instance_ids.shape or label_ids.ndim != 2:
        raise ValueError("Cityscapes labelIds and instanceIds must be matching 2D arrays")
    by_raw_id = {item.raw_id: item for item in CITYSCAPES_CLASSES}
    semantic = np.full(label_ids.shape, 255, dtype=np.uint8)
    instance = np.zeros(label_ids.shape, dtype=np.uint16)
    for raw_id in np.unique(label_ids):
        definition = by_raw_id.get(int(raw_id))
        selected = label_ids == raw_id
        if definition is None:
            if int(raw_id) == 255:
                continue
            if strict:
                raise ValueError(f"unknown Cityscapes label ID: {int(raw_id)}")
            continue
        if definition.train_id != 255:
            semantic[selected] = definition.train_id
    next_instance_id = 1
    for raw_id in np.unique(label_ids):
        definition = by_raw_id.get(int(raw_id))
        if definition is None or definition.train_id == 255 or not definition.has_instances:
            continue
        class_pixels = label_ids == raw_id
        raw_codes = np.unique(instance_ids[class_pixels])
        for raw_code in raw_codes:
            raw_code_int = int(raw_code)
            selected = class_pixels & (instance_ids == raw_code)
            if raw_code_int <= 0:
                if strict and bool(selected.any()):
                    raise ValueError(f"thing class {definition.name} contains a non-positive instance ID")
                continue
            if raw_code_int == definition.raw_id or raw_code_int % 1000 == 0:
                # Cityscapes uses a bare category ID (and some derived data
                # uses a zero suffix) for group/crowd regions. Schema v1 has
                # no crowd field, so exclude them from local supervision.
                semantic[selected] = 255
                continue
            encoded_category = raw_code_int // 1000
            if encoded_category != definition.raw_id:
                message = f"instance code {raw_code_int} has category {encoded_category}, expected {definition.raw_id}"
                if strict:
                    raise ValueError(message)
                continue
            if next_instance_id > np.iinfo(np.uint16).max:
                raise ValueError("more than 65535 thing instances in one image")
            instance[selected] = next_instance_id
            next_instance_id += 1
    return semantic, instance


def convert_cityscapes_dataset(
    data_root: str | Path,
    output_root: str | Path,
    *,
    copy_images: bool = True,
    strict: bool = True,
) -> Path:
    """Convert official Cityscapes train/val data into prepared project data."""
    source = Path(data_root).resolve()
    output = Path(output_root).resolve()
    left_root, gt_root = source / "leftImg8bit", source / "gtFine"
    if not left_root.is_dir() or not gt_root.is_dir():
        raise FileNotFoundError("Cityscapes root must contain leftImg8bit/ and gtFine/")
    schema = cityscapes_schema()
    for directory in ("images", "semantic", "instance"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, list[dict[str, str]]] = {"train": [], "valid": [], "test": []}
    panoptic_annotations: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    for source_split, target_split in (("train", "train"), ("val", "valid")):
        image_files = sorted((left_root / source_split).glob("*/*_leftImg8bit.png"))
        if not image_files:
            raise FileNotFoundError(f"no Cityscapes images found for split {source_split}")
        for image_path in image_files:
            city = image_path.parent.name
            base = image_path.name.removesuffix("_leftImg8bit.png")
            label_path = gt_root / source_split / city / f"{base}_gtFine_labelIds.png"
            instance_path = gt_root / source_split / city / f"{base}_gtFine_instanceIds.png"
            if not label_path.is_file() or not instance_path.is_file():
                raise FileNotFoundError(f"missing gtFine pair for {image_path}")
            sample_id = f"{source_split}__{city}__{base}"
            output_image = output / "images" / f"{sample_id}.png"
            output_semantic = output / "semantic" / f"{sample_id}.png"
            output_instance = output / "instance" / f"{sample_id}.png"
            output_panoptic = output / "panoptic" / target_split / f"{sample_id}.png"
            if copy_images:
                shutil.copy2(image_path, output_image)
            else:
                if output_image.exists() or output_image.is_symlink():
                    output_image.unlink()
                output_image.symlink_to(image_path)
            with Image.open(label_path) as opened:
                label_ids = np.asarray(opened, dtype=np.int64).copy()
            with Image.open(instance_path) as opened:
                instance_ids = np.asarray(opened, dtype=np.int64).copy()
            semantic, instance = convert_cityscapes_labels(label_ids, instance_ids, strict=strict)
            Image.fromarray(semantic).save(output_semantic)
            Image.fromarray(instance).save(output_instance)
            encoded = cityscapes_panoptic_ids(semantic, instance)
            write_cityscapes_panoptic_png(encoded, output_panoptic)
            panoptic_annotations[target_split].append(
                {
                    "image_id": base,
                    "file_name": output_panoptic.name,
                    "image_file_name": image_path.name,
                    "width": int(label_ids.shape[1]),
                    "height": int(label_ids.shape[0]),
                    "segments_info": cityscapes_segments_info(semantic, instance),
                }
            )
            split_rows[target_split].append(
                {
                    "sample_id": sample_id,
                    "provider_sample_id": base,
                    "source_split": source_split,
                    "image_path": os.path.relpath(output_image, output),
                    "semantic_path": os.path.relpath(output_semantic, output),
                    "instance_path": os.path.relpath(output_instance, output),
                }
            )
    _write_cityscapes_metadata(output, schema, split_rows, source, panoptic_annotations)
    return output


def _write_cityscapes_metadata(
    output: Path,
    schema: LabelSchema,
    split_rows: dict[str, list[dict[str, str]]],
    source: Path,
    panoptic_annotations: dict[str, list[dict[str, Any]]],
) -> None:
    schema_path = output / "schema.yaml"
    schema.write_yaml(schema_path)
    manifest_hashes: dict[str, str] = {}
    for split, rows in split_rows.items():
        path = output / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_id",
                    "provider_sample_id",
                    "source_split",
                    "image_path",
                    "semantic_path",
                    "instance_path",
                ),
            )
            writer.writeheader()
            writer.writerows(rows)
        manifest_hashes[split] = _sha256(path)
    schema_hash = _sha256(schema_path)
    identity_source = schema_hash + "".join(manifest_hashes[name] for name in ("train", "valid", "test"))
    metadata: dict[str, Any] = {
        "format_version": 1,
        "provider": "cityscapes",
        "source_root": str(source),
        "official_splits": {"train": "train", "valid": "val", "test": None},
        "test_available": False,
        "split_counts": {name: len(rows) for name, rows in split_rows.items()},
        "manifest_sha256": manifest_hashes,
        "schema_sha256": schema_hash,
        "identity": hashlib.sha256(identity_source.encode()).hexdigest(),
        "cityscapes_schema_version": CITYSCAPES_SCHEMA_VERSION,
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    for split in ("train", "valid"):
        write_cityscapes_panoptic_json(output / f"panoptic_{split}.json", panoptic_annotations[split])
    (output / "summary.txt").write_text(
        "\n".join(
            [
                "provider: cityscapes",
                f"identity: {metadata['identity']}",
                *[f"{name}: {len(rows)}" for name, rows in split_rows.items()],
                "test: unavailable because Cityscapes test annotations are not public",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
