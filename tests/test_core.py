from __future__ import annotations

import torch

from panoptic_segmenter.data.transforms import build_targets
from panoptic_segmenter.evaluation.metrics import panoptic_quality


def test_target_offsets_point_to_instance_center() -> None:
    semantic = torch.zeros((8, 8), dtype=torch.int64)
    semantic[2:6, 2:6] = 1
    instance = torch.zeros((8, 8), dtype=torch.int64)
    instance[2:6, 2:6] = 1
    target = build_targets(semantic, instance, (1,))
    assert int(target["center"].sum()) == 1
    assert torch.equal(target["offset"][:, 2, 2], torch.tensor([1.5, 1.5]))


def test_perfect_panoptic_quality() -> None:
    semantic = torch.tensor([[0, 1, 1], [0, 2, 2]])
    instance = torch.tensor([[0, 1, 1], [0, 0, 0]])
    scores = panoptic_quality(semantic, instance, semantic, instance, classes=((0, False), (1, True), (2, False)))
    assert scores["pq"] == 1.0
    assert scores["sq"] == 1.0
    assert scores["rq"] == 1.0
