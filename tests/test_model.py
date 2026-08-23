import torch

from panoptic_segmenter.models import available_models, create_model, register_model


def test_panoptic_model_contract() -> None:
    model = create_model("panoptic_unet_small", in_channels=3, num_classes=3, base_channels=4)
    output = model(torch.randn(1, 3, 64, 64))
    assert output["semantic"].shape == (1, 3, 64, 64)
    assert output["center"].shape == (1, 1, 64, 64)
    assert output["offset"].shape == (1, 2, 64, 64)


def test_model_registry_accepts_additional_factories() -> None:
    register_model("test_identity", lambda **_: torch.nn.Identity())
    assert "test_identity" in available_models()
    assert isinstance(create_model("test_identity", in_channels=3, num_classes=3, base_channels=4), torch.nn.Identity)
