"""Strict YAML configuration for reproducible experiments."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    data_dir: Path = Path("data/raw")
    manifest_dir: Path = Path("data/manifests")
    image_size: tuple[int, int] = (256, 256)
    batch_size: int = 4
    num_workers: int = 0
    horizontal_flip: float = 0.5
    center_sigma: float = 8.0
    max_train_samples: int | None = 256
    max_valid_samples: int | None = 64
    max_test_samples: int | None = None


@dataclass
class ModelConfig:
    name: str = "panoptic_unet_small"
    in_channels: int = 3
    expected_num_classes: int = 3
    base_channels: int = 32


@dataclass
class LossConfig:
    semantic_weight: float = 1.0
    center_weight: float = 1.0
    offset_weight: float = 0.01
    ignore_index: int = 255


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.1
    amp: bool = True
    grad_clip: float = 1.0
    seed: int = 42
    best_metric: str = "pq"


@dataclass
class PostprocessConfig:
    center_threshold: float = 0.2
    nms_kernel: int = 7
    top_k_centers: int = 200
    instance_area: int = 16
    stuff_area: int = 64


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    device: str = "auto"
    output_dir: Path = Path("artifacts")
    run_name: str = "panoptic-unet-small"


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    values = dataclasses.asdict(ExperimentConfig())
    _merge(values, raw)
    for key, value in (overrides or {}).items():
        if "." not in key:
            if key not in values or isinstance(values[key], dict):
                raise ValueError(f"unknown configuration field: {key}")
            values[key] = value
            continue
        section, field_name = key.split(".", 1)
        if section not in values or not isinstance(values[section], dict) or field_name not in values[section]:
            raise ValueError(f"unknown configuration field: {key}")
        values[section][field_name] = value
    values["data"]["data_dir"] = Path(values["data"]["data_dir"])
    values["data"]["manifest_dir"] = Path(values["data"]["manifest_dir"])
    values["output_dir"] = Path(values["output_dir"])
    values["data"]["image_size"] = tuple(values["data"]["image_size"])
    config = ExperimentConfig(
        data=DataConfig(**values["data"]),
        model=ModelConfig(**values["model"]),
        loss=LossConfig(**values["loss"]),
        train=TrainConfig(**values["train"]),
        postprocess=PostprocessConfig(**values["postprocess"]),
        device=values["device"],
        output_dir=values["output_dir"],
        run_name=values["run_name"],
    )
    _validate(config)
    return config


def _merge(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for key, value in incoming.items():
        if key not in target:
            raise ValueError(f"unknown configuration field: {key}")
        if isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a mapping")
            for child in value:
                if child not in target[key]:
                    raise ValueError(f"unknown configuration field: {key}.{child}")
            target[key].update(value)
        else:
            target[key] = value


def _validate(config: ExperimentConfig) -> None:
    if len(config.data.image_size) != 2 or any(int(value) <= 0 for value in config.data.image_size):
        raise ValueError("data.image_size must contain two positive integers")
    if any(int(value) % 16 for value in config.data.image_size):
        raise ValueError("data.image_size values must be divisible by 16")
    if config.data.batch_size < 1 or config.data.num_workers < 0:
        raise ValueError("data batch_size and num_workers are invalid")
    if not 0 <= config.data.horizontal_flip <= 1:
        raise ValueError("data.horizontal_flip must be between 0 and 1")
    if config.data.center_sigma <= 0:
        raise ValueError("data.center_sigma must be positive")
    for name in ("max_train_samples", "max_valid_samples", "max_test_samples"):
        value = getattr(config.data, name)
        if value is not None and value < 1:
            raise ValueError(f"data.{name} must be positive or null")
    if config.model.expected_num_classes < 2 or config.model.base_channels < 4:
        raise ValueError("model class count and base_channels are invalid")
    if any(value < 0 for value in (config.loss.semantic_weight, config.loss.center_weight, config.loss.offset_weight)):
        raise ValueError("loss weights must be non-negative")
    if config.train.epochs < 1 or config.train.lr <= 0 or config.train.weight_decay < 0:
        raise ValueError("train epochs, lr, and weight_decay are invalid")
    if config.train.optimizer not in {"adamw", "sgd"}:
        raise ValueError("train.optimizer must be adamw or sgd")
    if config.train.scheduler not in {"none", "cosine", "step"}:
        raise ValueError("train.scheduler must be none, cosine, or step")
    if config.train.scheduler_step_size < 1 or not 0 < config.train.scheduler_gamma <= 1:
        raise ValueError("train scheduler settings are invalid")
    if config.train.grad_clip < 0 or config.train.best_metric not in {"pq", "sq", "rq", "pq_thing", "pq_stuff"}:
        raise ValueError("train grad_clip or best_metric is invalid")
    postprocess = config.postprocess
    if not 0 <= postprocess.center_threshold <= 1:
        raise ValueError("postprocess.center_threshold must be between 0 and 1")
    if postprocess.nms_kernel < 1 or postprocess.nms_kernel % 2 == 0:
        raise ValueError("postprocess.nms_kernel must be a positive odd integer")
    if postprocess.top_k_centers < 1 or postprocess.instance_area < 0 or postprocess.stuff_area < 0:
        raise ValueError("postprocess center and area settings are invalid")
    if config.loss.ignore_index < 0:
        raise ValueError("loss.ignore_index must be non-negative")
    if config.device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")


def to_dict(config: ExperimentConfig) -> dict[str, Any]:
    value = dataclasses.asdict(config)

    def serialize(item: Any) -> Any:
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {key: serialize(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [serialize(child) for child in item]
        return item

    return serialize(value)
