"""Reproducible training, validation, resume, and artifact writing."""

from __future__ import annotations

import csv
import math
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ..config import ExperimentConfig, to_dict
from ..data import LabelSchema, PanopticDataset, inspect_prepared_dataset, panoptic_collate
from ..data.transforms import PanopticTransform
from ..evaluation.metrics import PanopticQualityAccumulator
from ..inference.postprocess import decode_panoptic
from ..models import create_model
from .checkpoint import build_run_metadata, capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from .losses import panoptic_loss

METRIC_FIELDS = (
    "epoch",
    "lr",
    "train_loss",
    "train_semantic",
    "train_center",
    "train_offset",
    "valid_loss",
    "pq",
    "sq",
    "rq",
    "pq_thing",
    "pq_stuff",
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _schema(config: ExperimentConfig) -> LabelSchema:
    path = config.data.manifest_dir / "schema.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"schema not found: {path}; run panoptic-segment prepare-data first")
    return LabelSchema.read_yaml(path)


def _loader(config: ExperimentConfig, schema: LabelSchema, split: str, training: bool) -> DataLoader:
    limits = {
        "train": config.data.max_train_samples,
        "valid": config.data.max_valid_samples,
        "test": config.data.max_test_samples,
    }
    if split not in limits:
        raise ValueError(f"unknown data split: {split}")
    dataset = PanopticDataset(
        config.data.manifest_dir / f"{split}.csv",
        PanopticTransform(
            config.data.image_size,
            config.data.horizontal_flip,
            training,
            config.data.center_sigma,
            schema.ignore_index,
        ),
        schema,
        limits[split],
    )
    generator = torch.Generator().manual_seed(config.train.seed)
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=training,
        num_workers=config.data.num_workers,
        collate_fn=panoptic_collate,
        generator=generator,
        pin_memory=config.device == "cuda",
        persistent_workers=config.data.num_workers > 0,
    )


def _optimizer(config: ExperimentConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if config.train.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=config.train.lr, momentum=0.9, weight_decay=config.train.weight_decay)


def _scheduler(
    config: ExperimentConfig, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.train.scheduler == "none":
        return None
    if config.train.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.train.scheduler_step_size, gamma=config.train.scheduler_gamma
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs)


def train_from_config(config: ExperimentConfig, *, dry_run: bool = False, resume: str | Path | None = None) -> Path:
    seed_everything(config.train.seed)
    device = resolve_device(config.device)
    schema = _schema(config)
    if schema.num_classes != config.model.expected_num_classes:
        raise ValueError("model.expected_num_classes does not match data schema")
    if schema.ignore_index != config.loss.ignore_index:
        raise ValueError("loss.ignore_index does not match data schema")
    data_report = inspect_prepared_dataset(config.data.manifest_dir)
    data_report.raise_for_issues()
    train_loader = _loader(config, schema, "train", True)
    valid_loader = _loader(config, schema, "valid", False)
    model = create_model(
        config.model.name,
        in_channels=config.model.in_channels,
        num_classes=schema.num_classes,
        base_channels=config.model.base_channels,
    ).to(device)
    optimizer = _optimizer(config, model)
    scheduler = _scheduler(config, optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=config.train.amp and device.type == "cuda")
    run_dir = config.output_dir / config.run_name
    if dry_run:
        _dry_run(model, train_loader, optimizer, scaler, device, config)
        return run_dir
    if resume is not None:
        resume_path = Path(resume).resolve()
        if resume_path.name != "last.pt" or resume_path.parent != run_dir.resolve():
            raise ValueError("resume must use last.pt from the configured run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    start_epoch, best, history = 1, -float("inf"), []
    dataset_metadata = yaml.safe_load((config.data.manifest_dir / "dataset.yaml").read_text(encoding="utf-8"))
    dataset_identity = str(dataset_metadata["identity"])
    if resume is not None:
        checkpoint = load_checkpoint(resume)
        _validate_resume(checkpoint, config, schema, dataset_identity)
        model.load_state_dict(_mapping(checkpoint["model_state"], "model_state"))
        optimizer.load_state_dict(dict(_mapping(checkpoint["optimizer_state"], "optimizer_state")))
        if scheduler is not None and isinstance(checkpoint.get("scheduler_state"), Mapping):
            scheduler.load_state_dict(dict(checkpoint["scheduler_state"]))
        if isinstance(checkpoint.get("scaler_state"), Mapping):
            scaler.load_state_dict(dict(checkpoint["scaler_state"]))
        if isinstance(checkpoint.get("rng_state"), Mapping):
            restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint["best_metric"])
        history = [dict(row) for row in checkpoint.get("metrics", []) if isinstance(row, Mapping)]
        if start_epoch > config.train.epochs:
            raise ValueError("train.epochs must be greater than the resumed checkpoint epoch")
    elif metrics_path.exists():
        raise FileExistsError(f"run already exists at {run_dir}; use --resume or a new run_name")
    resolved_config = to_dict(config)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8")
    run_metadata = build_run_metadata(device, config.train.seed)
    run_record = {
        **run_metadata,
        "dataset_identity": dataset_identity,
        "split_counts": data_report.split_counts,
        "started_at_unix": time.time(),
        "resumed_from": str(resume) if resume is not None else None,
    }
    (run_dir / "run.yaml").write_text(yaml.safe_dump(run_record, sort_keys=False), encoding="utf-8")
    mode = "a" if resume is not None and metrics_path.exists() else "w"
    with metrics_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if mode == "w":
            writer.writeheader()
        for epoch in range(start_epoch, config.train.epochs + 1):
            training = _train_epoch(model, train_loader, optimizer, scaler, device, config)
            valid = evaluate(model, valid_loader, schema, device, config)
            row: dict[str, float | int] = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                **training,
                **{
                    f"valid_{key}" if key == "loss" else key: valid[key]
                    for key in ("loss", "pq", "sq", "rq", "pq_thing", "pq_stuff")
                },
            }
            writer.writerow(row)
            handle.flush()
            history.append(row)
            metric_value = float(row[config.train.best_metric])
            if not math.isfinite(metric_value):
                raise RuntimeError(f"validation metric {config.train.best_metric} is non-finite")
            if scheduler is not None:
                scheduler.step()
            payload = _checkpoint_payload(
                config,
                schema,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                max(best, metric_value),
                history,
                run_metadata,
                dataset_identity,
            )
            save_checkpoint(run_dir / "last.pt", payload)
            if metric_value > best:
                best = metric_value
                save_checkpoint(run_dir / "best.pt", payload)
            print(
                f"epoch {epoch}/{config.train.epochs} loss={row['train_loss']:.4f} "
                f"pq={row['pq']:.4f} pq_th={row['pq_thing']:.4f} pq_st={row['pq_stuff']:.4f}"
            )
    run_record["completed_at_unix"] = time.time()
    run_record["completed_epochs"] = config.train.epochs
    run_record["best_metric"] = best
    (run_dir / "run.yaml").write_text(yaml.safe_dump(run_record, sort_keys=False), encoding="utf-8")
    return run_dir


def _dry_run(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: ExperimentConfig,
) -> None:
    images, targets, _ = next(iter(loader))
    moved = [{key: value.to(device) for key, value in target.items()} for target in targets]
    optimizer.zero_grad(set_to_none=True)
    outputs = model(images.to(device))
    loss, components = _loss(outputs, moved, config)
    if not torch.isfinite(loss):
        raise RuntimeError("dry-run produced a non-finite loss")
    scaler.scale(loss).backward()
    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    if config.train.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    print(f"dry-run OK images={tuple(images.shape)} loss={loss.item():.6f} components={components}")


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: ExperimentConfig,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "semantic": 0.0, "center": 0.0, "offset": 0.0}
    count = 0
    for images, targets, _ in loader:
        images = images.to(device)
        moved = [{key: value.to(device) for key, value in target.items()} for target in targets]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=config.train.amp and device.type == "cuda"):
            loss, components = _loss(model(images), moved, config)
        if not torch.isfinite(loss):
            raise RuntimeError("training produced a non-finite loss")
        scaler.scale(loss).backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        if config.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch = images.shape[0]
        totals["loss"] += float(loss.detach()) * batch
        for name in ("semantic", "center", "offset"):
            totals[name] += components[name] * batch
        count += batch
    if not count:
        raise ValueError("training loader yielded no samples")
    return {
        "train_loss": totals["loss"] / count,
        "train_semantic": totals["semantic"] / count,
        "train_center": totals["center"] / count,
        "train_offset": totals["offset"] / count,
    }


def evaluate(
    model: torch.nn.Module, loader: DataLoader, schema: LabelSchema, device: torch.device, config: ExperimentConfig
) -> dict[str, float]:
    model.eval()
    total_loss, count = 0.0, 0
    accumulator = PanopticQualityAccumulator(
        tuple((item.id, item.isthing) for item in schema.classes), schema.ignore_index
    )
    with torch.inference_mode():
        for images, targets, _ in loader:
            outputs = model(images.to(device))
            moved = [{key: value.to(device) for key, value in target.items()} for target in targets]
            loss, _ = _loss(outputs, moved, config)
            pred_semantic, pred_instance = decode_panoptic(
                outputs,
                schema.thing_ids,
                ignore_index=schema.ignore_index,
                **to_dict(config)["postprocess"],
            )
            for index, target in enumerate(moved):
                accumulator.update(pred_semantic[index], pred_instance[index], target["semantic"], target["instance"])
            total_loss += float(loss) * images.shape[0]
            count += images.shape[0]
    if not count:
        raise ValueError("evaluation loader yielded no samples")
    return {"loss": total_loss / count, **accumulator.compute()}


def _loss(
    outputs: dict[str, torch.Tensor], targets: list[dict[str, torch.Tensor]], config: ExperimentConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    return panoptic_loss(
        outputs,
        targets,
        semantic_weight=config.loss.semantic_weight,
        center_weight=config.loss.center_weight,
        offset_weight=config.loss.offset_weight,
        ignore_index=config.loss.ignore_index,
    )


def _checkpoint_payload(
    config: ExperimentConfig,
    schema: LabelSchema,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best: float,
    history: list[dict[str, Any]],
    run_metadata: dict[str, object],
    dataset_identity: str,
) -> dict[str, object]:
    return {
        "config": to_dict(config),
        "schema": schema.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_metric": best,
        "metrics": history,
        "run_metadata": run_metadata,
        "dataset_identity": dataset_identity,
        "rng_state": capture_rng_state(),
    }


def _validate_resume(
    checkpoint: dict[str, object], config: ExperimentConfig, schema: LabelSchema, dataset_identity: str
) -> None:
    if checkpoint.get("schema") != schema.to_dict():
        raise ValueError("resume schema does not match prepared data")
    if checkpoint.get("dataset_identity") != dataset_identity:
        raise ValueError("resume dataset identity does not match prepared manifests")
    saved = checkpoint.get("config")
    if not isinstance(saved, Mapping):
        raise ValueError("resume checkpoint config is invalid")
    current = to_dict(config)
    for section in ("model", "loss", "postprocess"):
        if saved.get(section) != current[section]:
            raise ValueError(f"resume {section} configuration does not match")
    saved_data = dict(_mapping(saved.get("data"), "config.data"))
    current_data = dict(current["data"])
    for name in ("data_dir", "manifest_dir", "num_workers"):
        saved_data.pop(name, None)
        current_data.pop(name, None)
    if saved_data != current_data:
        raise ValueError("resume data configuration does not match")
    saved_train = dict(_mapping(saved.get("train"), "config.train"))
    current_train = dict(current["train"])
    saved_train.pop("epochs", None)
    current_train.pop("epochs", None)
    if saved_train != current_train:
        raise ValueError("resume training configuration does not match")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {name} must be a mapping")
    return value
