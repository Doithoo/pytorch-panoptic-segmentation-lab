"""Run a complete reference job on a Kaggle T4-or-newer GPU.

Expected input layout: /kaggle/input/<dataset>/{images,semantic,instance}.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path

import torch

from panoptic_segmenter.config import load_config
from panoptic_segmenter.data import LabelSchema, default_label_schema, inspect_prepared_dataset
from panoptic_segmenter.data.manifest import prepare_paired_dataset
from panoptic_segmenter.evaluation.evaluate import evaluate_checkpoint
from panoptic_segmenter.training.checkpoint import sha256_file
from panoptic_segmenter.training.train import train_from_config


def _discover_input() -> Path:
    candidates = [
        path.parent
        for path in Path("/kaggle/input").rglob("images")
        if path.is_dir() and (path.parent / "semantic").is_dir() and (path.parent / "instance").is_dir()
    ]
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise SystemExit(f"expected exactly one panoptic dataset below /kaggle/input, found {unique}")
    return unique[0]


def _cuda_preflight() -> str:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; select a T4 or newer GPU in Kaggle settings")
    device_name = torch.cuda.get_device_name(0)
    try:
        value = torch.randn(32, 32, device="cuda", requires_grad=True)
        value.square().mean().backward()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise SystemExit(f"CUDA kernel preflight failed on {device_name}: {exc}") from exc
    return device_name


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--config", default="configs/reference_kaggle.yaml")
    parser.add_argument("--schema", default=None)
    parser.add_argument("--output", default="/kaggle/working/artifacts")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    started = time.time()
    gpu = _cuda_preflight()
    input_root = Path(args.input) if args.input else _discover_input()
    config = load_config(args.config)
    config.data.data_dir = input_root
    config.data.manifest_dir = Path("/kaggle/working/manifests")
    config.output_dir = Path(args.output)
    config.device = "cuda"
    schema = LabelSchema.read_yaml(args.schema) if args.schema else default_label_schema()
    prepare_paired_dataset(input_root, config.data.manifest_dir, schema=schema)
    report = inspect_prepared_dataset(config.data.manifest_dir)
    report.raise_for_issues()
    print(json.dumps({"phase": "preflight", "gpu": gpu, "input": str(input_root), "splits": report.split_counts}))
    run_dir = train_from_config(config, resume=args.resume)
    evaluation = evaluate_checkpoint(run_dir / "best.pt", split="test", device="cuda")
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    safe_evaluation = _json_safe(evaluation)
    (evaluation_dir / "evaluation.json").write_text(
        json.dumps(safe_evaluation, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (evaluation_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("class_id", "pq", "sq", "rq"))
        writer.writeheader()
        class_ids = sorted(int(key.split("class_", 1)[1]) for key in evaluation if key.startswith("pq:class_"))
        for class_id in class_ids:
            writer.writerow(
                {
                    "class_id": class_id,
                    "pq": _json_safe(evaluation.get(f"pq:class_{class_id}")),
                    "sq": _json_safe(evaluation.get(f"sq:class_{class_id}")),
                    "rq": _json_safe(evaluation.get(f"rq:class_{class_id}")),
                }
            )
    summary = {
        "status": "complete",
        "gpu": gpu,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "input": str(input_root),
        "split_counts": report.split_counts,
        "completed_epochs": config.train.epochs,
        "elapsed_seconds": time.time() - started,
        "best_checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "test": safe_evaluation,
    }
    summary_path = Path("/kaggle/working/kaggle-run-summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"phase": "complete", "run_dir": str(run_dir), "summary": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
