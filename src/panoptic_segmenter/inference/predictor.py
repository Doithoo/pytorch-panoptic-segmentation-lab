"""Checkpoint-backed inference for one image or a directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import ExperimentConfig, load_config
from ..data import LabelSchema
from ..data.transforms import PanopticTransform
from .postprocess import decode_panoptic


class Predictor:
    def __init__(self, model: torch.nn.Module, schema: LabelSchema, device: torch.device) -> None:
        self.model, self.schema, self.device = model.eval(), schema, device

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> Predictor:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = load_config_from_dict(checkpoint["config"])
        schema = LabelSchema.from_dict(checkpoint["schema"])
        from ..models import create_model

        resolved = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        )
        model = create_model(
            config.model.name,
            in_channels=config.model.in_channels,
            num_classes=schema.num_classes,
            base_channels=config.model.base_channels,
        )
        model.load_state_dict(checkpoint["model"])
        model.to(resolved)
        return cls(model, schema, resolved)

    def predict_path(self, image_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        transform = PanopticTransform((image.height // 16 * 16, image.width // 16 * 16), 0.0, False)
        tensor, _, _ = transform(image, Image.new("L", image.size), Image.new("I", image.size))
        with torch.inference_mode():
            semantic, instance = decode_panoptic(self.model(tensor[None].to(self.device)), self.schema.thing_ids)
        semantic_path = output / "semantic.png"
        instance_path = output / "instance.png"
        Image.fromarray(semantic[0].cpu().numpy().astype(np.uint8)).save(semantic_path)
        Image.fromarray(instance[0].cpu().numpy().astype(np.uint16)).save(instance_path)
        return {"semantic": semantic_path, "instance": instance_path}


def load_config_from_dict(raw: dict[str, object]) -> ExperimentConfig:
    import tempfile

    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as handle:
        handle.write(yaml.safe_dump(raw))
        handle.flush()
        return load_config(handle.name)
