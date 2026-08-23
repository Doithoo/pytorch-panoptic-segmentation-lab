"""Optional official-format panoptic evaluation adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def evaluate_with_panopticapi(
    ground_truth_json: str | Path,
    prediction_json: str | Path,
    ground_truth_folder: str | Path,
    prediction_folder: str | Path,
) -> dict[str, Any]:
    """Run the optional ``panopticapi`` evaluator on exported files.

    The dependency is deliberately optional: local training and the teaching
    evaluator do not require benchmark tooling. Install the pinned evaluator
    from the benchmark environment before calling this function.
    """
    try:
        from panopticapi.evaluation import pq_compute  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "panopticapi is required for official-format evaluation; install it in the benchmark environment"
        ) from exc
    result = pq_compute(
        str(ground_truth_json),
        str(prediction_json),
        str(ground_truth_folder),
        str(prediction_folder),
    )
    if not isinstance(result, dict):
        raise TypeError("panopticapi returned an unexpected result")
    return result
