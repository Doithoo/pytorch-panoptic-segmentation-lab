from __future__ import annotations

import torch

from panoptic_segmenter.data.transforms import build_targets
from panoptic_segmenter.evaluation.metrics import PanopticQualityAccumulator, panoptic_quality


def test_target_offsets_point_to_instance_center() -> None:
    semantic = torch.zeros((8, 8), dtype=torch.int64)
    semantic[2:6, 2:6] = 1
    instance = torch.zeros((8, 8), dtype=torch.int64)
    instance[2:6, 2:6] = 1
    target = build_targets(semantic, instance, (1,), center_sigma=1.0)
    assert float(target["center"].max()) > 0.75
    assert float(target["center"].sum()) > 1
    assert int(target["offset_mask"].sum()) == 16
    assert torch.equal(target["offset"][:, 2, 2], torch.tensor([1.5, 1.5]))


def test_offset_mask_excludes_stuff_and_ignore() -> None:
    semantic = torch.tensor([[0, 0, 255], [1, 1, 255]])
    instance = torch.tensor([[4, 4, 0], [2, 2, 0]])
    target = build_targets(semantic, instance, (1,), center_sigma=1.0, ignore_index=255)
    assert torch.equal(target["offset_mask"], torch.tensor([[False, False, False], [True, True, False]]))
    assert not bool(target["center_mask"][:, 2].any())


def test_perfect_panoptic_quality() -> None:
    semantic = torch.tensor([[0, 1, 1], [0, 2, 2]])
    instance = torch.tensor([[0, 1, 1], [0, 0, 0]])
    scores = panoptic_quality(semantic, instance, semantic, instance, classes=((0, False), (1, True), (2, False)))
    assert scores["pq"] == 1.0
    assert scores["sq"] == 1.0
    assert scores["rq"] == 1.0
    assert scores["pq_thing"] == 1.0
    assert scores["pq_stuff"] == 1.0


def test_predictions_inside_void_do_not_count_as_false_positives() -> None:
    target_semantic = torch.tensor([[0, 255], [0, 255]])
    target_instance = torch.zeros_like(target_semantic)
    pred_semantic = torch.tensor([[0, 1], [0, 1]])
    pred_instance = torch.tensor([[0, 1], [0, 1]])
    scores = panoptic_quality(
        pred_semantic,
        pred_instance,
        target_semantic,
        target_instance,
        classes=((0, False), (1, True)),
    )
    assert scores["pq"] == 1.0
    assert scores["fp"] == 0.0


def test_accumulator_macro_averages_classes_after_multiple_images() -> None:
    accumulator = PanopticQualityAccumulator(((0, False), (1, True)))
    target = torch.tensor([[0, 1], [0, 1]])
    target_instance = torch.tensor([[0, 1], [0, 1]])
    accumulator.update(target, target_instance, target, target_instance)
    wrong_instance = torch.zeros_like(target_instance)
    accumulator.update(target, wrong_instance, target, target_instance)
    scores = accumulator.compute()
    assert scores["pq:class_0"] == 1.0
    assert 0 < scores["pq:class_1"] < 1
    assert scores["pq"] == (scores["pq:class_0"] + scores["pq:class_1"]) / 2
