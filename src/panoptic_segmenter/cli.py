"""Command-line entry point."""

from __future__ import annotations

import argparse

import yaml

from .config import load_config, to_dict
from .data import LabelSchema
from .data.manifest import prepare_paired_dataset
from .evaluation.evaluate import evaluate_checkpoint
from .inference import Predictor
from .training.train import train_from_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="panoptic-segment")
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show-config")
    show.add_argument("--config", default=None)
    prepare = sub.add_parser("prepare-data")
    prepare.add_argument("--data-dir", default="data/raw")
    prepare.add_argument("--manifest-dir", default="data/manifests")
    prepare.add_argument("--schema", default="configs/cityscapes_mini_schema.yaml")
    train = sub.add_parser("train")
    train.add_argument("--config", default="configs/learning_minimal.yaml")
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    train.add_argument("--dry-run", action="store_true")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    evaluate.add_argument("--device", default="auto")
    predict = sub.add_parser("predict")
    predict.add_argument("checkpoint")
    predict.add_argument("image")
    predict.add_argument("--output", default="artifacts/prediction")
    predict.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    if args.command == "show-config":
        print(yaml.safe_dump(to_dict(load_config(args.config)), sort_keys=False), end="")
        return 0
    if args.command == "prepare-data":
        schema = LabelSchema.read_yaml(args.schema)
        paths = prepare_paired_dataset(args.data_dir, args.manifest_dir, schema=schema)
        for split, path in paths.items():
            print(f"{split}: {path}")
        return 0
    if args.command == "evaluate":
        print(yaml.safe_dump(evaluate_checkpoint(args.checkpoint, args.split, args.device), sort_keys=False), end="")
        return 0
    if args.command == "predict":
        paths = Predictor.from_checkpoint(args.checkpoint, args.device).predict_path(args.image, args.output)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return 0
    config = load_config(args.config)
    if args.device is not None:
        config.device = args.device
    print(f"run artifacts: {train_from_config(config, dry_run=args.dry_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
