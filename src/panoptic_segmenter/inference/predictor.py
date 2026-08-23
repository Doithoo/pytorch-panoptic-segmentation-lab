"""Safe checkpoint-backed inference with original-size outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import ExperimentConfig, load_config, to_dict
from ..data import LabelSchema
from ..data.transforms import PanopticTransform
from ..evaluation.visualization import colorize_semantic, panoptic_overlay
from ..training.checkpoint import load_checkpoint
from .postprocess import decode_panoptic


class Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        schema: LabelSchema,
        config: ExperimentConfig,
        device: torch.device,
    ) -> None:
        self.model, self.schema, self.config, self.device = model.eval(), schema, config, device

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> Predictor:
        checkpoint = load_checkpoint(path)
        config = load_config_from_dict(checkpoint["config"])
        schema = LabelSchema.from_dict(checkpoint["schema"])
        from ..models import create_model

        if device == "auto":
            resolved = torch.device(
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if resolved.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        model = create_model(
            config.model.name,
            in_channels=config.model.in_channels,
            num_classes=schema.num_classes,
            base_channels=config.model.base_channels,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(resolved)
        return cls(model, schema, config, resolved)

    def predict_path(self, image_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        transform = PanopticTransform(self.config.data.image_size, 0.0, False)
        tensor, _, _ = transform(image, Image.new("L", image.size), Image.new("I", image.size))
        with torch.inference_mode():
            semantic, instance = decode_panoptic(
                self.model(tensor[None].to(self.device)),
                self.schema.thing_ids,
                ignore_index=self.schema.ignore_index,
                **to_dict(self.config)["postprocess"],
            )
        semantic_array = semantic[0].cpu().numpy().astype(np.uint8)
        instance_array = instance[0].cpu().numpy().astype(np.uint16)
        semantic_array = np.asarray(
            Image.fromarray(semantic_array).resize(image.size, Image.Resampling.NEAREST), dtype=np.uint8
        )
        instance_array = np.asarray(
            Image.fromarray(instance_array).resize(image.size, Image.Resampling.NEAREST), dtype=np.uint16
        )
        stem = Path(image_path).stem
        semantic_path = output / f"{stem}.semantic.png"
        instance_path = output / f"{stem}.instance.png"
        color_path = output / f"{stem}.semantic-color.png"
        overlay_path = output / f"{stem}.overlay.png"
        Image.fromarray(semantic_array.astype(np.uint8)).save(semantic_path)
        Image.fromarray(instance_array.astype(np.uint16)).save(instance_path)
        colorize_semantic(semantic_array, self.schema).save(color_path)
        panoptic_overlay(image, semantic_array, instance_array, self.schema).save(overlay_path)
        return {
            "semantic": semantic_path,
            "instance": instance_path,
            "semantic_color": color_path,
            "overlay": overlay_path,
        }


def load_config_from_dict(raw: dict[str, object]) -> ExperimentConfig:
    import tempfile

    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as handle:
        handle.write(yaml.safe_dump(raw))
        handle.flush()
        return load_config(handle.name)
