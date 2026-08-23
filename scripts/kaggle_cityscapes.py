"""Train and officially evaluate an attached licensed Cityscapes dataset."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

from panoptic_segmenter.config import load_config
from panoptic_segmenter.data import convert_cityscapes_dataset, discover_cityscapes_root, inspect_prepared_dataset
from panoptic_segmenter.evaluation.cityscapes import (
    create_official_cityscapes_ground_truth,
    evaluate_with_cityscapesscripts,
)
from panoptic_segmenter.evaluation.evaluate import evaluate_checkpoint
from panoptic_segmenter.training.checkpoint import sha256_file
from panoptic_segmenter.training.train import train_from_config


def cuda_preflight() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable a T4 or newer GPU")
    device_name = torch.cuda.get_device_name(0)
    value = torch.randn(32, 32, device="cuda", requires_grad=True)
    value.square().mean().backward()
    torch.cuda.synchronize()
    return device_name


def json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/kaggle/input")
    parser.add_argument("--config", default="configs/cityscapes.yaml")
    parser.add_argument("--output", default="/kaggle/working/artifacts")
    parser.add_argument("--prepared", default="/kaggle/working/cityscapes-prepared")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    started = time.time()
    gpu = cuda_preflight()
    source = discover_cityscapes_root(args.input)
    prepared = convert_cityscapes_dataset(source, args.prepared, copy_images=False)
    report = inspect_prepared_dataset(prepared)
    report.raise_for_issues()
    config = load_config(args.config)
    config.data.data_dir = prepared
    config.data.manifest_dir = prepared
    config.output_dir = Path(args.output)
    config.device = "cuda"
    run_dir = train_from_config(config, resume=args.resume)
    internal = evaluate_checkpoint(run_dir / "best.pt", split="valid", device="cuda")
    predictions = Path("/kaggle/working/cityscapes-predictions")
    subprocess.run(
        [
            sys.executable,
            "scripts/predict_cityscapes.py",
            str(run_dir / "best.pt"),
            "--manifest",
            str(prepared / "valid.csv"),
            "--output",
            str(predictions),
            "--device",
            "cuda",
        ],
        check=True,
    )
    official_root = Path("/kaggle/working/cityscapes-official-panoptic")
    ground_truth_json, ground_truth_folder = create_official_cityscapes_ground_truth(source, official_root, split="val")
    official = evaluate_with_cityscapesscripts(
        ground_truth_json,
        predictions / "predictions.json",
        ground_truth_folder,
        predictions / "panoptic",
        run_dir / "evaluation/cityscapes-official.json",
    )
    summary = {
        "status": "complete",
        "protocol": "cityscapes-official-val",
        "gpu": gpu,
        "source_root": str(source),
        "split_counts": report.split_counts,
        "elapsed_seconds": time.time() - started,
        "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "internal_validation": json_safe(internal),
        "official_validation": json_safe(official),
    }
    summary_path = Path("/kaggle/working/cityscapes-run-summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"phase": "complete", "summary": str(summary_path), "run_dir": str(run_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
