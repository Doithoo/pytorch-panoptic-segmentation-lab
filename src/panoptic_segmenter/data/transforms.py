"""Synchronized image and panoptic-label transforms."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image


class PanopticTransform:
    def __init__(
        self, image_size: tuple[int, int] = (256, 256), horizontal_flip: float = 0.5, training: bool = False
    ) -> None:
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.training = training

    def __call__(
        self, image: Image.Image, semantic: Image.Image, instance: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = image.resize((self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR)
        semantic = semantic.resize((self.image_size[1], self.image_size[0]), Image.Resampling.NEAREST)
        instance = instance.resize((self.image_size[1], self.image_size[0]), Image.Resampling.NEAREST)
        if self.training and random.random() < self.horizontal_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            semantic = semantic.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            instance = instance.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
        image_tensor = (image_tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None]
        return (
            image_tensor,
            torch.from_numpy(np.asarray(semantic, dtype=np.int64)),
            torch.from_numpy(np.asarray(instance, dtype=np.int64)),
        )


def build_targets(
    semantic: torch.Tensor, instance: torch.Tensor, thing_ids: tuple[int, ...]
) -> dict[str, torch.Tensor]:
    """Create Panoptic-DeepLab style center and offset supervision."""
    center = torch.zeros_like(semantic, dtype=torch.float32)
    offset = torch.zeros((2, *semantic.shape), dtype=torch.float32)
    for instance_id in torch.unique(instance).tolist():
        if instance_id <= 0:
            continue
        pixels = instance == instance_id
        if not pixels.any():
            continue
        class_ids = semantic[pixels]
        if int(class_ids[0]) not in thing_ids:
            continue
        ys, xs = torch.where(pixels)
        cy, cx = ys.float().mean(), xs.float().mean()
        distance = (ys.float() - cy).square() + (xs.float() - cx).square()
        center[ys[distance.argmin()], xs[distance.argmin()]] = 1.0
        offset[0, ys, xs] = cy - ys.float()
        offset[1, ys, xs] = cx - xs.float()
    return {"semantic": semantic, "instance": instance, "center": center, "offset": offset}
