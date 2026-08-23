"""Adapters for the official ``cityscapesscripts`` panoptic workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def create_official_cityscapes_ground_truth(
    cityscapes_root: str | Path,
    output_root: str | Path,
    *,
    split: str = "val",
) -> tuple[Path, Path]:
    """Generate crowd-aware panoptic GT through the official package."""
    if split not in {"train", "val"}:
        raise ValueError("official local panoptic ground truth supports train or val")
    try:
        from cityscapesscripts.preparation.createPanopticImgs import convert2panoptic  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("cityscapesscripts is required for official Cityscapes ground truth") from exc
    source = Path(cityscapes_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    convert2panoptic(
        cityscapesPath=str(source / "gtFine"),
        outputFolder=str(output),
        useTrainId=False,
        setNames=[split],
    )
    json_path = output / f"cityscapes_panoptic_{split}.json"
    folder = output / f"cityscapes_panoptic_{split}"
    if not json_path.is_file() or not folder.is_dir():
        raise RuntimeError("cityscapesscripts did not create expected panoptic ground truth")
    return json_path, folder


def evaluate_with_cityscapesscripts(
    ground_truth_json: str | Path,
    prediction_json: str | Path,
    ground_truth_folder: str | Path,
    prediction_folder: str | Path,
    results_path: str | Path,
) -> dict[str, Any]:
    """Evaluate predictions with the official Cityscapes panoptic evaluator."""
    try:
        from cityscapesscripts.evaluation.evalPanopticSemanticLabeling import (  # type: ignore[import-not-found]
            evaluatePanoptic,
        )
    except ImportError as exc:
        raise RuntimeError("cityscapesscripts is required for official Cityscapes evaluation") from exc
    destination = Path(results_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = evaluatePanoptic(
        str(ground_truth_json),
        str(ground_truth_folder),
        str(prediction_json),
        str(prediction_folder),
        str(destination),
    )
    if not isinstance(result, dict):
        if not destination.is_file():
            raise TypeError("cityscapesscripts returned an unexpected result")
        loaded = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise TypeError("Cityscapes result file must contain a mapping")
        return loaded
    return result
