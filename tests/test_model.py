import torch

from panoptic_segmenter.models import create_model


def test_panoptic_model_contract() -> None:
    model = create_model("panoptic_unet_small", in_channels=3, num_classes=3, base_channels=4)
    output = model(torch.randn(1, 3, 64, 64))
    assert output["semantic"].shape == (1, 3, 64, 64)
    assert output["center"].shape == (1, 1, 64, 64)
    assert output["offset"].shape == (1, 2, 64, 64)
