"""Evaluate a saved panoptic checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from ..data import LabelSchema
from ..inference.predictor import load_config_from_dict
from ..models import create_model
from ..training.train import _loader, evaluate, resolve_device


def evaluate_checkpoint(checkpoint_path: str | Path, split: str = "valid", device: str = "auto") -> dict[str, float]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = load_config_from_dict(checkpoint["config"])
    schema = LabelSchema.from_dict(checkpoint["schema"])
    resolved = resolve_device(device)
    model = create_model(
        config.model.name,
        in_channels=config.model.in_channels,
        num_classes=schema.num_classes,
        base_channels=config.model.base_channels,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(resolved)
    return evaluate(model, _loader(config, schema, split, False), schema, resolved, config)
