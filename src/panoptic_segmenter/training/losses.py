"""Losses for semantic, center, and offset branches."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
    semantic_loss = F.cross_entropy(outputs["semantic"], semantic, ignore_index=ignore_index)
    center_loss = F.binary_cross_entropy_with_logits(outputs["center"][:, 0], center)
    valid = (torch.stack([target["instance"] for target in targets]) > 0).to(outputs["offset"].device)
    if valid.any():
        offset_loss = F.l1_loss(outputs["offset"].permute(0, 2, 3, 1)[valid], offset.permute(0, 2, 3, 1)[valid])
    else:
        offset_loss = outputs["offset"].sum() * 0.0
    total = semantic_weight * semantic_loss + center_weight * center_loss + offset_weight * offset_loss
    return total, {
        "semantic": float(semantic_loss.detach()),
        "center": float(center_loss.detach()),
        "offset": float(offset_loss.detach()),
    }
