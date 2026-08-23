"""Evaluate exported Cityscapes-compatible panoptic files."""

from __future__ import annotations

import argparse
import json

from panoptic_segmenter.evaluation.panopticapi import evaluate_with_panopticapi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-json", required=True)
    parser.add_argument("--prediction-json", required=True)
    parser.add_argument("--ground-truth-folder", required=True)
    parser.add_argument("--prediction-folder", required=True)
    args = parser.parse_args()
    scores = evaluate_with_panopticapi(
        args.ground_truth_json,
        args.prediction_json,
        args.ground_truth_folder,
        args.prediction_folder,
    )
    print(json.dumps(scores, indent=2, allow_nan=False))
