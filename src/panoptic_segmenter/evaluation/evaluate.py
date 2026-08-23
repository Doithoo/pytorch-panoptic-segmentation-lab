"""Evaluate a saved panoptic checkpoint."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from ..data import LabelSchema, inspect_prepared_dataset
from ..inference.postprocess import decode_panoptic
from ..inference.predictor import load_config_from_dict
from ..models import create_model
from ..training.checkpoint import load_checkpoint, sha256_file
from ..training.train import _loader, evaluate, resolve_device
from .metrics import PanopticQualityAccumulator, panoptic_quality

_IMAGE_METRICS = ("pq", "sq", "rq", "pq_thing", "pq_stuff", "tp", "fp", "fn")


def _load_context(
    checkpoint_path: str | Path, device: str
) -> tuple[dict[str, Any], Any, LabelSchema, torch.device, torch.nn.Module]:
    checkpoint = load_checkpoint(checkpoint_path)
    config = load_config_from_dict(checkpoint["config"])
    schema = LabelSchema.from_dict(checkpoint["schema"])
    report = inspect_prepared_dataset(config.data.manifest_dir)
    report.raise_for_issues()
    metadata = yaml.safe_load((config.data.manifest_dir / "dataset.yaml").read_text(encoding="utf-8"))
    checkpoint_identity = checkpoint.get("dataset_identity")
    if checkpoint_identity is not None and metadata.get("identity") != checkpoint_identity:
        raise ValueError("evaluation dataset identity does not match checkpoint")
    resolved = resolve_device(device)
    model = create_model(
        config.model.name,
        in_channels=config.model.in_channels,
        num_classes=schema.num_classes,
        base_channels=config.model.base_channels,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(resolved)
    return checkpoint, config, schema, resolved, model


def evaluate_checkpoint(checkpoint_path: str | Path, split: str = "valid", device: str = "auto") -> dict[str, float]:
    """Evaluate one split and return aggregate metrics."""
    _, config, schema, resolved, model = _load_context(checkpoint_path, device)
    return evaluate(model, _loader(config, schema, split, False), schema, resolved, config)


def evaluate_checkpoint_detailed(
    checkpoint_path: str | Path, split: str = "valid", device: str = "auto"
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Evaluate one split and return aggregate plus per-image metrics."""
    _, config, schema, resolved, model = _load_context(checkpoint_path, device)
    loader = _loader(config, schema, split, False)
    classes = tuple((item.id, item.isthing) for item in schema.classes)
    accumulator = PanopticQualityAccumulator(classes, schema.ignore_index)
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for images, targets, sample_ids in loader:
            outputs = model(images.to(resolved))
            predictions = decode_panoptic(
                outputs,
                schema.thing_ids,
                ignore_index=schema.ignore_index,
                **config.postprocess.__dict__,
            )
            pred_semantic, pred_instance = predictions
            moved = [{key: value.to(resolved) for key, value in target.items()} for target in targets]
            for index, sample_id in enumerate(sample_ids):
                image_scores = panoptic_quality(
                    pred_semantic[index],
                    pred_instance[index],
                    moved[index]["semantic"],
                    moved[index]["instance"],
                    classes=classes,
                    ignore_index=schema.ignore_index,
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        **{key: image_scores[key] for key in _IMAGE_METRICS},
                    }
                )
                accumulator.update(
                    pred_semantic[index],
                    pred_instance[index],
                    moved[index]["semantic"],
                    moved[index]["instance"],
                )
    return accumulator.compute(), rows


def _json_metrics(values: dict[str, float]) -> dict[str, float | None]:
    return {key: value if math.isfinite(float(value)) else None for key, value in values.items()}


def _row_metrics(row: Mapping[str, object]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in _IMAGE_METRICS:
        value = row.get(key)
        if not isinstance(value, (int, float)):
            raise TypeError(f"per-image metric {key!r} must be numeric")
        values[key] = float(value)
    return values


def write_evaluation_report(
    path: str | Path,
    checkpoint_path: str | Path,
    split: str,
    device: str,
    scores: dict[str, float],
    *,
    per_image: list[dict[str, object]] | None = None,
    worst_case_count: int = 10,
) -> Path:
    """Write auditable metadata, aggregate scores, and optional per-image metrics."""
    if worst_case_count < 0:
        raise ValueError("worst_case_count must be non-negative")
    checkpoint = Path(checkpoint_path).resolve()
    loaded = load_checkpoint(checkpoint)
    config = load_config_from_dict(loaded["config"])
    metadata = yaml.safe_load((config.data.manifest_dir / "dataset.yaml").read_text(encoding="utf-8"))
    rows = per_image or []
    ordered = sorted(rows, key=lambda row: _row_metrics(row)["pq"])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split": split,
        "requested_device": device,
        "resolved_device": str(resolve_device(device)),
        "dataset_identity": metadata["identity"],
        "metrics": _json_metrics(scores),
        "per_image": [{"sample_id": str(row["sample_id"]), **_json_metrics(_row_metrics(row))} for row in rows],
        "worst_cases": [
            {"sample_id": str(row["sample_id"]), **_json_metrics(_row_metrics(row))}
            for row in ordered[:worst_case_count]
        ],
    }
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return destination
