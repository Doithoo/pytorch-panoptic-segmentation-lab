"""Bounded panoptic post-processing from semantic, center, and offset outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def decode_panoptic(
    outputs: dict[str, torch.Tensor],
    thing_ids: tuple[int, ...],
    *,
    ignore_index: int = 255,
    center_threshold: float = 0.2,
    nms_kernel: int = 7,
    top_k_centers: int = 200,
    instance_area: int = 16,
    stuff_area: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode a batch while bounding center-assignment memory.

    Centers are selected only on predicted thing pixels. Thing pixels are
    assigned to a center of the same semantic class, and tiny thing/stuff
    regions become void.
    """
    if nms_kernel < 1 or nms_kernel % 2 == 0:
        raise ValueError("nms_kernel must be a positive odd integer")
    if top_k_centers < 1:
        raise ValueError("top_k_centers must be positive")
    semantic = outputs["semantic"].argmax(dim=1)
    center_scores = outputs["center"].sigmoid()[:, 0]
    pooled = F.max_pool2d(center_scores[:, None], nms_kernel, stride=1, padding=nms_kernel // 2)[:, 0]
    thing_mask = torch.zeros_like(semantic, dtype=torch.bool)
    for class_id in thing_ids:
        thing_mask |= semantic == class_id
    candidates = (center_scores >= center_threshold) & (center_scores == pooled) & thing_mask
    result_semantic = semantic.clone()
    result_instance = torch.zeros_like(semantic)
    for batch_index in range(semantic.shape[0]):
        selected = torch.where(candidates[batch_index].flatten())[0]
        if selected.numel() > top_k_centers:
            scores = center_scores[batch_index].flatten()[selected]
            selected = selected[scores.topk(top_k_centers).indices]
        center_y = torch.div(selected, semantic.shape[2], rounding_mode="floor")
        center_x = selected % semantic.shape[2]
        next_instance_id = 1
        for class_id in thing_ids:
            pixel_y, pixel_x = torch.where(semantic[batch_index] == class_id)
            class_centers = semantic[batch_index, center_y, center_x] == class_id
            class_center_y, class_center_x = center_y[class_centers], center_x[class_centers]
            if pixel_y.numel() == 0:
                continue
            if class_center_y.numel() == 0:
                result_semantic[batch_index, pixel_y, pixel_x] = ignore_index
                continue
            assignments: list[torch.Tensor] = []
            for start in range(0, pixel_y.numel(), 32768):
                ys = pixel_y[start : start + 32768]
                xs = pixel_x[start : start + 32768]
                predicted_y = ys + outputs["offset"][batch_index, 0, ys, xs]
                predicted_x = xs + outputs["offset"][batch_index, 1, ys, xs]
                distances = (predicted_y[:, None] - class_center_y).square() + (
                    predicted_x[:, None] - class_center_x
                ).square()
                assignments.append(distances.argmin(dim=1))
            assigned = torch.cat(assignments)
            for center_index in range(class_center_y.numel()):
                member = assigned == center_index
                if int(member.sum()) < instance_area:
                    result_semantic[batch_index, pixel_y[member], pixel_x[member]] = ignore_index
                    continue
                result_instance[batch_index, pixel_y[member], pixel_x[member]] = next_instance_id
                next_instance_id += 1
        for class_id in torch.unique(semantic[batch_index]).tolist():
            if class_id in thing_ids:
                continue
            area = semantic[batch_index] == class_id
            if int(area.sum()) < stuff_area:
                result_semantic[batch_index, area] = ignore_index
    return result_semantic, result_instance
