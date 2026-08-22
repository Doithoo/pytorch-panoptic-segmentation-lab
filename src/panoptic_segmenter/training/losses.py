"""Losses for semantic, center, and offset branches."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _center_focal_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
    positive = (target >= 0.999) & valid
    negative = (~positive) & valid
    negative_weight = (1 - target).pow(4)
    positive_loss = -torch.log(probability) * (1 - probability).pow(2) * positive
    negative_loss = -torch.log(1 - probability) * probability.pow(2) * negative_weight * negative
    positive_count = positive.sum().clamp_min(1)
    negative_count = negative.sum().clamp_min(1)
    return positive_loss.sum() / positive_count + negative_loss.sum() / negative_count


def panoptic_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    semantic_weight: float,
    center_weight: float,
    offset_weight: float,
    ignore_index: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    semantic = torch.stack([target["semantic"] for target in targets]).to(outputs["semantic"].device)
    center = torch.stack([target["center"] for target in targets]).to(outputs["center"].device)
    offset = torch.stack([target["offset"] for target in targets]).to(outputs["offset"].device)
    center_mask = torch.stack([target["center_mask"] for target in targets]).to(outputs["center"].device)
    offset_mask = torch.stack([target["offset_mask"] for target in targets]).to(outputs["offset"].device)
    semantic_loss = F.cross_entropy(outputs["semantic"], semantic, ignore_index=ignore_index)
    center_loss = _center_focal_loss(outputs["center"][:, 0], center, center_mask)
    if offset_mask.any():
        predicted = outputs["offset"].permute(0, 2, 3, 1)[offset_mask]
        expected = offset.permute(0, 2, 3, 1)[offset_mask]
        offset_loss = F.l1_loss(predicted, expected)
    else:
        offset_loss = outputs["offset"].sum() * 0.0
    total = semantic_weight * semantic_loss + center_weight * center_loss + offset_weight * offset_loss
    return total, {
        "semantic": float(semantic_loss.detach()),
        "center": float(center_loss.detach()),
        "offset": float(offset_loss.detach()),
    }
