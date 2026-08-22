"""Panoptic Quality metrics implemented without external evaluation packages."""

from __future__ import annotations

import math

import torch


def _segments(semantic: torch.Tensor, instance: torch.Tensor, class_id: int, thing: bool) -> list[torch.Tensor]:
    valid = semantic == class_id
    ids = torch.unique(instance[valid]) if thing else torch.tensor([0], device=semantic.device)
    return [
        valid & (instance == int(item))
        for item in ids
        if int(item) > 0 or not thing
        if bool((valid & (instance == int(item))).any())
    ]


def panoptic_quality(
    pred_semantic: torch.Tensor,
    pred_instance: torch.Tensor,
    target_semantic: torch.Tensor,
    target_instance: torch.Tensor,
    *,
    classes: tuple[tuple[int, bool], ...],
    ignore_index: int = 255,
) -> dict[str, float]:
    if pred_semantic.shape != target_semantic.shape or pred_instance.shape != target_instance.shape:
        raise ValueError("prediction and target shapes must match")
    totals = {"tp": 0, "fp": 0, "fn": 0, "iou": 0.0}
    class_scores: dict[str, float] = {}
    for class_id, isthing in classes:
        pred = _segments(pred_semantic, pred_instance, class_id, isthing)
        target = _segments(target_semantic, target_instance, class_id, isthing)
        candidates: list[tuple[float, int, int]] = []
        for pred_index, pred_mask in enumerate(pred):
            for target_index, target_mask in enumerate(target):
                intersection = (pred_mask & target_mask & (target_semantic != ignore_index)).sum().item()
                union = (pred_mask | target_mask).sum().item() - (
                    pred_mask & target_mask & (target_semantic == ignore_index)
                ).sum().item()
                if union:
                    iou = intersection / union
                    if iou > 0.5:
                        candidates.append((iou, pred_index, target_index))
        matched_pred, matched_target = set(), set()
        score = 0.0
        for iou, pred_index, target_index in sorted(candidates, reverse=True):
            if pred_index in matched_pred or target_index in matched_target:
                continue
            matched_pred.add(pred_index)
            matched_target.add(target_index)
            score += iou
        tp, fp, fn = len(matched_pred), len(pred) - len(matched_pred), len(target) - len(matched_target)
        denom = tp + 0.5 * fp + 0.5 * fn
        class_scores[str(class_id)] = score / denom if denom else math.nan
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["iou"] += score
    denom = totals["tp"] + 0.5 * totals["fp"] + 0.5 * totals["fn"]
    return {
        "pq": totals["iou"] / denom if denom else 0.0,
        "sq": totals["iou"] / totals["tp"] if totals["tp"] else 0.0,
        "rq": totals["tp"] / denom if denom else 0.0,
        "tp": float(totals["tp"]),
        "fp": float(totals["fp"]),
        "fn": float(totals["fn"]),
        **{f"pq:class_{key}": value for key, value in class_scores.items()},
    }
