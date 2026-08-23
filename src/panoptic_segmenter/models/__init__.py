"""Extensible panoptic model registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from .panoptic_unet import PanopticUNet

ModelFactory = Callable[..., nn.Module]
_MODEL_FACTORIES: dict[str, ModelFactory] = {}


def register_model(name: str, factory: ModelFactory, *, overwrite: bool = False) -> None:
    """Register a model factory that accepts the standard keyword arguments."""
    if not name.strip():
        raise ValueError("model name must be non-empty")
    if name in _MODEL_FACTORIES and not overwrite:
        raise ValueError(f"model is already registered: {name}")
    _MODEL_FACTORIES[name] = factory


def create_model(name: str, *, in_channels: int, num_classes: int, base_channels: int) -> nn.Module:
    try:
        factory = _MODEL_FACTORIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown panoptic model {name!r}; available: {available_models()}") from exc
    model = factory(in_channels=in_channels, num_classes=num_classes, base_channels=base_channels)
    if not isinstance(model, nn.Module):
        raise TypeError(f"model factory {name!r} returned {type(model).__name__}, not torch.nn.Module")
    return model


def available_models() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_FACTORIES))


def _panoptic_unet_factory(*, in_channels: int, num_classes: int, base_channels: int, **_: Any) -> PanopticUNet:
    return PanopticUNet(in_channels, num_classes, base_channels)


register_model("panoptic_unet_small", _panoptic_unet_factory)
