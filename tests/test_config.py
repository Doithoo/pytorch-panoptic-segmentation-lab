from __future__ import annotations

import pytest

from panoptic_segmenter.config import load_config


def test_reference_config_loads() -> None:
    config = load_config("configs/learning_minimal.yaml")
    assert config.model.name == "panoptic_unet_small"
    assert config.data.image_size == (256, 256)
    assert config.postprocess.top_k_centers == 100


def test_defaults_load_without_repository_config_path() -> None:
    config = load_config()
    assert config.train.optimizer == "adamw"
    assert config.train.scheduler == "cosine"


def test_top_level_and_nested_overrides_are_typed() -> None:
    config = load_config(overrides={"run_name": "changed", "data.image_size": [32, 48]})
    assert config.run_name == "changed"
    assert config.data.image_size == (32, 48)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"train.optimizer": "invalid"}, "optimizer"),
        ({"postprocess.nms_kernel": 4}, "odd"),
        ({"data.image_size": [31, 32]}, "divisible"),
        ({"loss.center_weight": -1}, "non-negative"),
    ],
)
def test_invalid_configuration_is_rejected(override: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_config(overrides=override)
