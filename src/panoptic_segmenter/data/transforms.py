"""Synchronized image and panoptic-label transforms."""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image


class PanopticTransform:
    def __init__(
        self,
        image_size: tuple[int, int] = (256, 256),
        horizontal_flip: float = 0.5,
        training: bool = False,
        center_sigma: float = 8.0,
        ignore_index: int = 255,
    ) -> None:
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.training = training
        self.center_sigma = center_sigma
        self.ignore_index = ignore_index

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
    semantic: torch.Tensor,
    instance: torch.Tensor,
    thing_ids: tuple[int, ...],
    *,
    center_sigma: float = 8.0,
    ignore_index: int = 255,
) -> dict[str, torch.Tensor]:
    """Create Gaussian center heatmaps and thing-only offset supervision."""
    if semantic.shape != instance.shape or semantic.ndim != 2:
        raise ValueError("semantic and instance masks must be matching two-dimensional tensors")
    center = torch.zeros_like(semantic, dtype=torch.float32)
    offset = torch.zeros((2, *semantic.shape), dtype=torch.float32)
    offset_mask = torch.zeros_like(semantic, dtype=torch.bool)
    valid_mask = semantic != ignore_index
    radius = max(1, int(round(3 * center_sigma)))
    for instance_id in torch.unique(instance).tolist():
        if instance_id <= 0:
            continue
        pixels = (instance == instance_id) & valid_mask
        if not pixels.any():
            continue
        class_ids = torch.unique(semantic[pixels])
        if class_ids.numel() != 1 or int(class_ids[0]) not in thing_ids:
            continue
        ys, xs = torch.where(pixels)
        cy, cx = ys.float().mean(), xs.float().mean()
        y0, y1 = max(0, int(cy) - radius), min(semantic.shape[0], int(cy) + radius + 1)
        x0, x1 = max(0, int(cx) - radius), min(semantic.shape[1], int(cx) + radius + 1)
        grid_y = torch.arange(y0, y1, dtype=torch.float32, device=semantic.device)
        grid_x = torch.arange(x0, x1, dtype=torch.float32, device=semantic.device)
        gaussian = torch.exp(
            -((grid_y[:, None] - cy).square() + (grid_x[None, :] - cx).square()) / (2 * center_sigma**2)
        )
        center[y0:y1, x0:x1] = torch.maximum(center[y0:y1, x0:x1], gaussian)
        nearest_y = min(semantic.shape[0] - 1, max(0, int(torch.round(cy).item())))
        nearest_x = min(semantic.shape[1] - 1, max(0, int(torch.round(cx).item())))
        center[nearest_y, nearest_x] = 1.0
        offset[0, ys, xs] = cy - ys.float()
        offset[1, ys, xs] = cx - xs.float()
        offset_mask[pixels] = True
    return {
        "semantic": semantic,
        "instance": instance,
        "center": center,
        "offset": offset,
        "center_mask": valid_mask,
        "offset_mask": offset_mask,
    }
