"""Model registry."""

from .panoptic_unet import PanopticUNet


def create_model(name: str, *, in_channels: int, num_classes: int, base_channels: int) -> PanopticUNet:
    if name != "panoptic_unet_small":
        raise ValueError(f"unknown panoptic model: {name}")
    return PanopticUNet(in_channels, num_classes, base_channels)


def available_models() -> tuple[str, ...]:
    return ("panoptic_unet_small",)
