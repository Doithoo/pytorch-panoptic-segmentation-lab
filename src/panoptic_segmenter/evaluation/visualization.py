"""Schema-driven semantic colors and panoptic overlays."""

from __future__ import annotations

import colorsys

import numpy as np
from PIL import Image

from ..data import LabelSchema

IGNORE_COLOR = (128, 128, 128)


def colorize_semantic(mask: np.ndarray, schema: LabelSchema) -> Image.Image:
    if mask.ndim != 2:
        raise ValueError("semantic mask must be two-dimensional")
    allowed = set(range(schema.num_classes)) | {schema.ignore_index}
    unexpected = sorted({int(value) for value in np.unique(mask)} - allowed)
    if unexpected:
        raise ValueError(f"semantic mask contains unsupported IDs: {unexpected}")
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for item in schema.classes:
        result[mask == item.id] = item.color
    result[mask == schema.ignore_index] = IGNORE_COLOR
    return Image.fromarray(result)


def panoptic_overlay(
    image: Image.Image,
    semantic: np.ndarray,
    instance: np.ndarray,
    schema: LabelSchema,
    alpha: float = 0.5,
) -> Image.Image:
    if semantic.shape != instance.shape or semantic.ndim != 2:
        raise ValueError("semantic and instance masks must have matching two-dimensional shapes")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    source = np.asarray(image.convert("RGB"), dtype=np.float32)
    if source.shape[:2] != semantic.shape:
        raise ValueError("image and panoptic masks must have matching dimensions")
    colors = np.asarray(colorize_semantic(semantic, schema), dtype=np.float32)
    for instance_id in np.unique(instance[instance > 0]):
        selected = instance == instance_id
        class_id = int(np.bincount(semantic[selected].astype(np.int64)).argmax())
        colors[selected] = _instance_color(class_id, int(instance_id))
    selected = semantic != schema.ignore_index
    result = source.copy()
    result[selected] = (1 - alpha) * source[selected] + alpha * colors[selected]
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _instance_color(class_id: int, instance_id: int) -> tuple[int, int, int]:
    hue = ((class_id * 53 + instance_id * 97) % 360) / 360
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return round(red * 255), round(green * 255), round(blue * 255)
