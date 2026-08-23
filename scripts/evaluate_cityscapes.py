"""Create official GT or evaluate Cityscapes panoptic predictions."""

from __future__ import annotations

import argparse
import json

from panoptic_segmenter.evaluation.cityscapes import (
    create_official_cityscapes_ground_truth,
    evaluate_with_cityscapesscripts,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-ground-truth")
    prepare.add_argument("--cityscapes-root", required=True)
    prepare.add_argument("--output", default="data/cityscapes-official-panoptic")
    prepare.add_argument("--split", choices=("train", "val"), default="val")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--ground-truth-json", required=True)
    evaluate.add_argument("--prediction-json", required=True)
    evaluate.add_argument("--ground-truth-folder", required=True)
    evaluate.add_argument("--prediction-folder", required=True)
    evaluate.add_argument("--results", default="artifacts/cityscapes-official-results.json")
    args = parser.parse_args()
    if args.command == "prepare-ground-truth":
        json_path, folder = create_official_cityscapes_ground_truth(args.cityscapes_root, args.output, split=args.split)
        print(json.dumps({"json": str(json_path), "folder": str(folder)}, indent=2))
    else:
        result = evaluate_with_cityscapesscripts(
            args.ground_truth_json,
            args.prediction_json,
            args.ground_truth_folder,
            args.prediction_folder,
            args.results,
        )
        print(json.dumps(result, indent=2, allow_nan=False))
