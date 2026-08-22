from panoptic_segmenter.config import load_config


def test_reference_config_loads() -> None:
    config = load_config("configs/learning_minimal.yaml")
    assert config.model.name == "panoptic_unet_small"
    assert config.data.image_size == (256, 256)
