from __future__ import annotations

import numpy as np

from panoptic_segmenter.data import available_converters, rasterize_soccer_annotations, soccer_schema


def test_soccer_schema_and_polygon_rasterization() -> None:
    schema = soccer_schema()
    assert "kaggle-soccer" in available_converters()
    assert schema.num_classes == 7
    assert schema.thing_ids == (0, 1, 5)
    semantic, instance = rasterize_soccer_annotations(
        8,
        6,
        [
            {"category_id": 5, "segmentation": [[0, 0, 7, 0, 7, 5, 0, 5]]},
            {"category_id": 1, "segmentation": [[1, 1, 3, 1, 3, 3, 1, 3]]},
            {"category_id": 2, "segmentation": [[5, 2, 6, 2, 6, 3, 5, 3]]},
        ],
        {1: 0, 2: 1, 5: 4},
    )
    assert semantic[2, 2] == 0
    assert semantic[2, 5] == 1
    assert semantic[0, 0] == 4
    assert np.unique(instance).tolist() == [0, 1, 2]
    assert np.all(instance[semantic == 1] == 2)


def test_soccer_polygon_rasterization_rejects_rle() -> None:
    try:
        rasterize_soccer_annotations(4, 4, [{"category_id": 1, "segmentation": {"counts": []}}], {1: 0})
    except ValueError as exc:
        assert "polygon" in str(exc)
    else:
        raise AssertionError("expected polygon-only validation to fail")
