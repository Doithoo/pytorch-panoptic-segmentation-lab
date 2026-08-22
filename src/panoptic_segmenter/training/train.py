"""Readable training loop with checkpoint and metrics artifacts."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig, to_dict
from ..data import LabelSchema, PanopticDataset, panoptic_collate
from ..data.transforms import PanopticTransform
from ..evaluation.metrics import panoptic_quality
from ..inference.postprocess import decode_panoptic
from ..models import create_model
from .losses import panoptic_loss


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def _schema(config: ExperimentConfig) -> LabelSchema:
    path = config.data.manifest_dir / "schema.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"schema not found: {path}; run panoptic-segment prepare-data first")
    return LabelSchema.read_yaml(path)


def _loader(config: ExperimentConfig, schema: LabelSchema, split: str, training: bool) -> DataLoader:
    limit = config.data.max_train_samples if split == "train" else config.data.max_valid_samples
    dataset = PanopticDataset(
        config.data.manifest_dir / f"{split}.csv",
        PanopticTransform(config.data.image_size, config.data.horizontal_flip, training),
        schema,
        limit,
    )
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=training,
        num_workers=config.data.num_workers,
        collate_fn=panoptic_collate,
    )


def train_from_config(config: ExperimentConfig, *, dry_run: bool = False) -> Path:
    seed_everything(config.train.seed)
    device = resolve_device(config.device)
    schema = _schema(config)
    if schema.num_classes != config.model.expected_num_classes:
        raise ValueError("model.expected_num_classes does not match data schema")
    train_loader = _loader(config, schema, "train", True)
    valid_loader = _loader(config, schema, "valid", False)
    model = create_model(
        config.model.name,
        in_channels=config.model.in_channels,
        num_classes=schema.num_classes,
        base_channels=config.model.base_channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.train.amp and device.type == "cuda")
    run_dir = config.output_dir / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        __import__("yaml").safe_dump(to_dict(config), sort_keys=False), encoding="utf-8"
    )
    if dry_run:
        images, targets, _ = next(iter(train_loader))
        outputs = model(images.to(device))
        loss, components = panoptic_loss(
            outputs,
            [{key: value.to(device) for key, value in target.items()} for target in targets],
            semantic_weight=config.loss.semantic_weight,
            center_weight=config.loss.center_weight,
            offset_weight=config.loss.offset_weight,
            ignore_index=config.loss.ignore_index,
        )
        print(
            f"images={tuple(images.shape)} semantic={tuple(outputs['semantic'].shape)} loss={loss.item():.6f} components={components}"
        )
        return run_dir
    best = -float("inf")
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "valid_loss", "pq", "sq", "rq"))
        writer.writeheader()
        for epoch in range(1, config.train.epochs + 1):
            model.train()
            train_total = 0.0
            train_count = 0
            for images, targets, _ in train_loader:
                images = images.to(device)
                moved = [{key: value.to(device) for key, value in target.items()} for target in targets]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, enabled=config.train.amp and device.type in {"cuda", "cpu"}
                ):
                    loss, _ = panoptic_loss(
                        model(images),
                        moved,
                        semantic_weight=config.loss.semantic_weight,
                        center_weight=config.loss.center_weight,
                        offset_weight=config.loss.offset_weight,
                        ignore_index=config.loss.ignore_index,
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_total += loss.item() * images.shape[0]
                train_count += images.shape[0]
            valid = evaluate(model, valid_loader, schema, device, config)
            row = {
                "epoch": epoch,
                "train_loss": train_total / train_count,
                "valid_loss": valid["loss"],
                **{key: valid[key] for key in ("pq", "sq", "rq")},
            }
            writer.writerow(row)
            handle.flush()
            print(f"epoch {epoch}/{config.train.epochs} loss={row['train_loss']:.4f} pq={row['pq']:.4f}")
            torch.save(
                {"model": model.state_dict(), "config": to_dict(config), "schema": schema.to_dict(), "epoch": epoch},
                run_dir / "last.pt",
            )
            if row["pq"] > best:
                best = row["pq"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": to_dict(config),
                        "schema": schema.to_dict(),
                        "epoch": epoch,
                    },
                    run_dir / "best.pt",
                )
    return run_dir


def evaluate(
    model: torch.nn.Module, loader: DataLoader, schema: LabelSchema, device: torch.device, config: ExperimentConfig
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    aggregate = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "iou": 0.0}
    with torch.inference_mode():
        for images, targets, _ in loader:
            outputs = model(images.to(device))
            moved = [{key: value.to(device) for key, value in target.items()} for target in targets]
            loss, _ = panoptic_loss(
                outputs,
                moved,
                semantic_weight=config.loss.semantic_weight,
                center_weight=config.loss.center_weight,
                offset_weight=config.loss.offset_weight,
                ignore_index=config.loss.ignore_index,
            )
            pred_semantic, pred_instance = decode_panoptic(outputs, schema.thing_ids)
            for index, target in enumerate(moved):
                scores = panoptic_quality(
                    pred_semantic[index],
                    pred_instance[index],
                    target["semantic"],
                    target["instance"],
                    classes=tuple((item.id, item.isthing) for item in schema.classes),
                    ignore_index=schema.ignore_index,
                )
                aggregate["tp"] += scores["tp"]
                aggregate["fp"] += scores["fp"]
                aggregate["fn"] += scores["fn"]
                aggregate["iou"] += scores["sq"] * scores["tp"]
            total_loss += loss.item() * images.shape[0]
            count += images.shape[0]
    denominator = aggregate["tp"] + 0.5 * aggregate["fp"] + 0.5 * aggregate["fn"]
    return {
        "loss": total_loss / count,
        "pq": aggregate["iou"] / denominator if denominator else 0.0,
        "sq": aggregate["iou"] / aggregate["tp"] if aggregate["tp"] else 0.0,
        "rq": aggregate["tp"] / denominator if denominator else 0.0,
    }
