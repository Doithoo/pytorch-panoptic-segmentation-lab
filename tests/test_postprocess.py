from __future__ import annotations

import torch

from panoptic_segmenter.inference.postprocess import decode_panoptic


def _outputs() -> dict[str, torch.Tensor]:
    semantic = torch.full((1, 3, 8, 8), -10.0)
    semantic[:, 0] = 10
    semantic[:, 0, 2:6, 2:6] = -10
    semantic[:, 1, 2:6, 2:6] = 10
    center = torch.full((1, 1, 8, 8), -10.0)
    offset = torch.zeros((1, 2, 8, 8))
    return {"semantic": semantic, "center": center, "offset": offset}


def test_decode_assigns_thing_pixels_to_bounded_centers() -> None:
    outputs = _outputs()
    outputs["center"][0, 0, 3, 3] = 10
    outputs["center"][0, 0, 4, 4] = 9
    semantic, instance = decode_panoptic(
        outputs,
        (1,),
        nms_kernel=1,
        top_k_centers=1,
        instance_area=1,
        stuff_area=1,
    )
    assert torch.all(semantic[0, 2:6, 2:6] == 1)
    assert set(torch.unique(instance).tolist()) == {0, 1}


def test_decode_turns_thing_without_center_into_void() -> None:
    semantic, instance = decode_panoptic(_outputs(), (1,), instance_area=1, stuff_area=1)
    assert torch.all(semantic[0, 2:6, 2:6] == 255)
    assert not bool(instance.any())


def test_decode_rejects_even_nms_kernel() -> None:
    try:
        decode_panoptic(_outputs(), (1,), nms_kernel=4)
    except ValueError as exc:
        assert "odd" in str(exc)
    else:
        raise AssertionError("expected invalid NMS kernel to fail")
