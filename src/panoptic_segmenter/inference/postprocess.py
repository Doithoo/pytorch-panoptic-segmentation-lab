"""Panoptic post-processing from semantic, center, and offset outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def decode_panoptic(
    outputs: dict[str, torch.Tensor], thing_ids: tuple[int, ...], *, center_threshold: float = 0.2, nms_kernel: int = 7
) -> tuple[torch.Tensor, torch.Tensor]:
    semantic = outputs["semantic"].argmax(dim=1)
    center_scores = outputs["center"].sigmoid()
    pooled = F.max_pool2d(center_scores, nms_kernel, stride=1, padding=nms_kernel // 2)
    centers = (center_scores >= center_threshold) & (center_scores == pooled)
    results_semantic, results_instance = [], []
    thing_mask_classes = torch.zeros_like(semantic, dtype=torch.bool)
    for class_id in thing_ids:
        thing_mask_classes |= semantic == class_id
    for batch_index in range(semantic.shape[0]):
        ys, xs = torch.where(centers[batch_index, 0])
        if ys.numel() == 0:
            results_semantic.append(semantic[batch_index])
            results_instance.append(torch.zeros_like(semantic[batch_index]))
            continue
        yy, xx = torch.meshgrid(
            torch.arange(semantic.shape[1], device=semantic.device),
            torch.arange(semantic.shape[2], device=semantic.device),
            indexing="ij",
        )
        predicted_center_y = yy + outputs["offset"][batch_index, 0]
        predicted_center_x = xx + outputs["offset"][batch_index, 1]
        distances = (predicted_center_y[..., None] - ys).square() + (predicted_center_x[..., None] - xs).square()
        nearest = distances.argmin(dim=-1) + 1
        instance = torch.where(thing_mask_classes[batch_index], nearest, torch.zeros_like(nearest))
        results_semantic.append(semantic[batch_index])
        results_instance.append(instance)
    return torch.stack(results_semantic), torch.stack(results_instance)
