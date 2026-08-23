"""Convert the public Kaggle Soccer dataset to the project mask contract."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .schema import ClassDefinition, LabelSchema

SOCCER_CLASS_NAMES = ("Player", "Ball", "Goal Line", "Field", "Background", "Referee", "Football Pitch Line")
SOCCER_THING_NAMES = frozenset(("Player", "Ball", "Referee"))
SOCCER_FRAME_RE = re.compile(r"_(\d+)\.[^.]+$")


def soccer_schema() -> LabelSchema:
    """Return the seven-class teaching schema used by the public dataset."""
    colors = ((210, 2, 27), (80, 227, 194), (189, 16, 224), (139, 87, 42), (0, 0, 0), (245, 166, 35), (74, 144, 226))
    return LabelSchema(
        classes=tuple(
            ClassDefinition(index, name, name in SOCCER_THING_NAMES, colors[index])
            for index, name in enumerate(SOCCER_CLASS_NAMES)
        ),
        ignore_index=255,
    )


def rasterize_soccer_annotations(
    width: int,
    height: int,
    annotations: list[dict[str, Any]],
    category_to_class: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize COCO polygon annotations into semantic and image-local instances."""
    semantic = Image.new("L", (width, height), color=SOCCER_CLASS_NAMES.index("Background"))
    instance = Image.new("I", (width, height), color=0)
    semantic_draw = ImageDraw.Draw(semantic)
    instance_draw = ImageDraw.Draw(instance)
    thing_classes = {SOCCER_CLASS_NAMES.index(name) for name in SOCCER_THING_NAMES}
    next_instance = 1
    for annotation in annotations:
        try:
            class_id = category_to_class[int(annotation["category_id"])]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"unknown soccer annotation category: {annotation.get('category_id')!r}") from exc
        segmentation = annotation.get("segmentation")
        if not isinstance(segmentation, list) or any(not isinstance(polygon, list) for polygon in segmentation):
            raise ValueError("soccer converter supports polygon segmentations only")
        polygons: list[list[tuple[int, int]]] = []
        for polygon in segmentation:
            if len(polygon) < 6 or len(polygon) % 2:
                raise ValueError("soccer polygon must contain at least three x/y pairs")
            polygons.append(
                [
                    (round(float(polygon[index])), round(float(polygon[index + 1])))
                    for index in range(0, len(polygon), 2)
                ]
            )
        for polygon in polygons:
            semantic_draw.polygon(polygon, fill=class_id)
            instance_draw.polygon(polygon, fill=next_instance if class_id in thing_classes else 0)
        if class_id in thing_classes:
            next_instance += 1
    return np.asarray(semantic, dtype=np.uint8), np.asarray(instance, dtype=np.uint16)


def convert_kaggle_soccer_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    max_frames: int | None = None,
    frame_stride: int = 1,
    resize_width: int | None = None,
    jpeg_quality: int = 90,
) -> Path:
    """Extract annotated soccer frames and write the project's three-folder contract.

    OpenCV is imported only when conversion runs, keeping the base package light.
    """
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive or null")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if resize_width is not None and resize_width < 16:
        raise ValueError("resize_width must be at least 16")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("opencv-python is required; run with `uv run --with opencv-python-headless`") from exc

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    annotation_files = sorted(source.rglob("instances.json"))
    if not annotation_files:
        raise FileNotFoundError(f"no instances.json files found below {source}")
    schema = soccer_schema()
    schema.write_yaml(output / "schema.yaml")
    extracted = 0
    group_rows: list[dict[str, str]] = []
    for annotation_path in annotation_files:
        video_dir = annotation_path.parent.parent / "video"
        videos = sorted(video_dir.glob("*.mp4"))
        if len(videos) != 1:
            raise ValueError(f"expected one video beside {annotation_path}, found {videos}")
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
        category_to_class = {}
        for category_id, name in categories.items():
            if name not in SOCCER_CLASS_NAMES:
                raise ValueError(f"unsupported soccer category: {name}")
            category_to_class[category_id] = SOCCER_CLASS_NAMES.index(name)
        annotations_by_image: dict[int, list[dict[str, Any]]] = {}
        for annotation in payload.get("annotations", []):
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        images = sorted(payload.get("images", []), key=lambda item: int(item["id"]))
        capture = cv2.VideoCapture(str(videos[0]))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open soccer video: {videos[0]}")
        try:
            for position, image_info in enumerate(images):
                if position % frame_stride:
                    continue
                if max_frames is not None and extracted >= max_frames:
                    break
                match = SOCCER_FRAME_RE.search(str(image_info["file_name"]))
                if match is None:
                    raise ValueError(f"cannot parse frame number from {image_info['file_name']!r}")
                frame_index = int(match.group(1))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                success, frame = capture.read()
                if not success:
                    raise RuntimeError(f"cannot read frame {frame_index} from {videos[0]}")
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = rgb.shape[:2]
                semantic, instance = rasterize_soccer_annotations(
                    width,
                    height,
                    annotations_by_image.get(int(image_info["id"]), []),
                    category_to_class,
                )
                image = Image.fromarray(rgb, mode="RGB")
                if resize_width is not None:
                    resize_height = round(height * resize_width / width)
                    image = image.resize((resize_width, resize_height), Image.Resampling.BILINEAR)
                    semantic = np.asarray(
                        Image.fromarray(semantic).resize((resize_width, resize_height), Image.Resampling.NEAREST)
                    )
                    instance = np.asarray(
                        Image.fromarray(instance).resize((resize_width, resize_height), Image.Resampling.NEAREST)
                    )
                sample_id = f"soccer_{int(image_info['id']):08d}"
                (output / "images").mkdir(parents=True, exist_ok=True)
                (output / "semantic").mkdir(parents=True, exist_ok=True)
                (output / "instance").mkdir(parents=True, exist_ok=True)
                image.save(output / "images" / f"{sample_id}.jpg", quality=jpeg_quality)
                Image.fromarray(semantic.astype(np.uint8)).save(output / "semantic" / f"{sample_id}.png")
                Image.fromarray(instance.astype(np.uint16)).save(output / "instance" / f"{sample_id}.png")
                group_rows.append({"sample_id": sample_id, "group_id": videos[0].stem})
                extracted += 1
        finally:
            capture.release()
        if max_frames is not None and extracted >= max_frames:
            break
    with (output / "groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "group_id"))
        writer.writeheader()
        writer.writerows(group_rows)
    (output / "source.json").write_text(
        json.dumps(
            {
                "dataset": "quantigoai/soccer-dataset",
                "license": "CC-BY-SA-4.0",
                "annotation_files": len(annotation_files),
                "frame_stride": frame_stride,
                "max_frames": max_frames,
                "resize_width": resize_width,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output
