"""Class-wise Panoptic Quality for the project's non-crowd mask contract."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass
class _ClassStat:
    iou: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def denominator(self) -> float:
        return self.tp + 0.5 * self.fp + 0.5 * self.fn


class PanopticQualityAccumulator:
    """Accumulate segment matches by class before macro averaging."""

    def __init__(self, classes: tuple[tuple[int, bool], ...], ignore_index: int = 255) -> None:
        self.classes = classes
        self.ignore_index = ignore_index
        self.stats = {class_id: _ClassStat() for class_id, _ in classes}

    def update(
        self,
        pred_semantic: torch.Tensor,
        pred_instance: torch.Tensor,
        target_semantic: torch.Tensor,
        target_instance: torch.Tensor,
    ) -> None:
        if pred_semantic.shape != target_semantic.shape or pred_instance.shape != target_instance.shape:
            raise ValueError("prediction and target shapes must match")
        if pred_semantic.ndim != 2:
            raise ValueError("panoptic masks must be two-dimensional")
        void = target_semantic == self.ignore_index
        for class_id, isthing in self.classes:
            pred = _segments(pred_semantic, pred_instance, class_id, isthing)
            target = _segments(target_semantic, target_instance, class_id, isthing)
            candidates: list[tuple[float, int, int]] = []
            for pred_index, pred_mask in enumerate(pred):
                pred_valid = pred_mask & ~void
                for target_index, target_mask in enumerate(target):
                    intersection = int((pred_valid & target_mask).sum())
                    union = int(pred_valid.sum()) + int(target_mask.sum()) - intersection
                    if union and intersection / union > 0.5:
                        candidates.append((intersection / union, pred_index, target_index))
            matched_pred: set[int] = set()
            matched_target: set[int] = set()
            stat = self.stats[class_id]
            for iou, pred_index, target_index in sorted(candidates, reverse=True):
                if pred_index in matched_pred or target_index in matched_target:
                    continue
                matched_pred.add(pred_index)
                matched_target.add(target_index)
                stat.iou += iou
            stat.tp += len(matched_pred)
            stat.fn += len(target) - len(matched_target)
            for pred_index, pred_mask in enumerate(pred):
                if pred_index in matched_pred:
                    continue
                area = int(pred_mask.sum())
                void_overlap = int((pred_mask & void).sum())
                if area and void_overlap / area <= 0.5:
                    stat.fp += 1

    def compute(self) -> dict[str, float]:
        class_results: dict[int, dict[str, float]] = {}
        for class_id, _ in self.classes:
            stat = self.stats[class_id]
            class_results[class_id] = {
                "pq": stat.iou / stat.denominator if stat.denominator else math.nan,
                "sq": stat.iou / stat.tp if stat.tp else math.nan,
                "rq": stat.tp / stat.denominator if stat.denominator else math.nan,
            }
        summary = {
            "pq": _mean(class_results[class_id]["pq"] for class_id, _ in self.classes),
            "sq": _mean(class_results[class_id]["sq"] for class_id, _ in self.classes),
            "rq": _mean(class_results[class_id]["rq"] for class_id, _ in self.classes),
            "pq_thing": _mean(class_results[class_id]["pq"] for class_id, thing in self.classes if thing),
            "pq_stuff": _mean(class_results[class_id]["pq"] for class_id, thing in self.classes if not thing),
            "tp": float(sum(stat.tp for stat in self.stats.values())),
            "fp": float(sum(stat.fp for stat in self.stats.values())),
            "fn": float(sum(stat.fn for stat in self.stats.values())),
        }
        for class_id, _ in self.classes:
            for metric, value in class_results[class_id].items():
                summary[f"{metric}:class_{class_id}"] = value
        return summary


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else 0.0


def _segments(semantic: torch.Tensor, instance: torch.Tensor, class_id: int, thing: bool) -> list[torch.Tensor]:
    valid = semantic == class_id
    if not thing:
        return [valid] if bool(valid.any()) else []
    return [valid & (instance == value) for value in torch.unique(instance[valid]).tolist() if int(value) > 0]


def panoptic_quality(
    pred_semantic: torch.Tensor,
    pred_instance: torch.Tensor,
    target_semantic: torch.Tensor,
    target_instance: torch.Tensor,
    *,
    classes: tuple[tuple[int, bool], ...],
    ignore_index: int = 255,
) -> dict[str, float]:
    accumulator = PanopticQualityAccumulator(classes, ignore_index)
    accumulator.update(pred_semantic, pred_instance, target_semantic, target_instance)
    return accumulator.compute()
