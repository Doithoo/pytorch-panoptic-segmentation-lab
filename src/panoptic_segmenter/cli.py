"""Command-line entry point."""

from __future__ import annotations

import argparse
from typing import Any

import yaml

from . import __version__
from .config import load_config, to_dict
from .data import (
    LabelSchema,
    cityscapes_schema,
    convert_cityscapes_dataset,
    default_label_schema,
    inspect_prepared_dataset,
)
from .data.manifest import prepare_paired_dataset
from .evaluation.evaluate import evaluate_checkpoint
from .inference import Predictor
from .training.train import train_from_config


def _overrides(values: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"override must use KEY=VALUE: {item}")
        key, raw = item.split("=", 1)
        result[key] = yaml.safe_load(raw)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="panoptic-segment")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    show = sub.add_parser("show-config", help="print the fully resolved configuration")
    show.add_argument("--config", default=None)
    show.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    prepare = sub.add_parser("prepare-data", help="pair source masks and create deterministic manifests")
    prepare.add_argument("--data-dir", default="data/raw")
    prepare.add_argument("--manifest-dir", default="data/manifests")
    prepare.add_argument("--schema", default=None)
    prepare.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    prepare.add_argument("--seed", type=int, default=42)
    city = sub.add_parser("convert-cityscapes", help="convert official Cityscapes train/val labels")
    city.add_argument("--data-root", required=True)
    city.add_argument("--output-root", default="data/cityscapes")
    city.add_argument("--symlink-images", action="store_true")
    city.add_argument("--non-strict", action="store_true")
    inspect = sub.add_parser("inspect-data", help="validate prepared manifests and panoptic labels")
    inspect.add_argument("--manifest-dir", default="data/manifests")
    inspect.add_argument("--limit-per-split", type=int, default=None)
    train = sub.add_parser("train", help="train, dry-run, or resume an experiment")
    train.add_argument("--config", default=None)
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    train.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    train.add_argument("--resume", default=None)
    train.add_argument("--dry-run", action="store_true")
    evaluate = sub.add_parser("evaluate", help="evaluate a safe checkpoint")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    predict = sub.add_parser("predict", help="export raw masks and visualizations for one image")
    predict.add_argument("checkpoint")
    predict.add_argument("image")
    predict.add_argument("--output", default="artifacts/prediction")
    predict.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args(argv)
    if args.command == "show-config":
        print(yaml.safe_dump(to_dict(load_config(args.config, _overrides(args.set))), sort_keys=False), end="")
        return 0
    if args.command == "prepare-data":
        schema = LabelSchema.read_yaml(args.schema) if args.schema else default_label_schema()
        paths = prepare_paired_dataset(
            args.data_dir,
            args.manifest_dir,
            schema=schema,
            ratios=tuple(args.ratios),
            seed=args.seed,
        )
        for split, path in paths.items():
            print(f"{split}: {path}")
        return 0
    if args.command == "convert-cityscapes":
        output = convert_cityscapes_dataset(
            args.data_root,
            args.output_root,
            copy_images=not args.symlink_images,
            strict=not args.non_strict,
        )
        print(f"cityscapes data: {output}")
        print(yaml.safe_dump(cityscapes_schema().to_dict(), sort_keys=False), end="")
        return 0
    if args.command == "inspect-data":
        report = inspect_prepared_dataset(args.manifest_dir, limit_per_split=args.limit_per_split)
        print(
            yaml.safe_dump(
                {
                    "split_counts": report.split_counts,
                    "inspected_samples": report.inspected_samples,
                    "issues": [vars(item) for item in report.issues],
                },
                sort_keys=False,
            ),
            end="",
        )
        report.raise_for_issues()
        return 0
    if args.command == "evaluate":
        print(yaml.safe_dump(evaluate_checkpoint(args.checkpoint, args.split, args.device), sort_keys=False), end="")
        return 0
    if args.command == "predict":
        paths = Predictor.from_checkpoint(args.checkpoint, args.device).predict_path(args.image, args.output)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return 0
    config = load_config(args.config, _overrides(args.set))
    if args.device is not None:
        config.device = args.device
    print(f"run artifacts: {train_from_config(config, dry_run=args.dry_run, resume=args.resume)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
