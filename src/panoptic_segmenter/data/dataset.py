"""Manifest-backed panoptic dataset.

Each CSV row contains sample_id,image_path,semantic_path,instance_path. Paths are
relative to the manifest directory unless absolute.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .schema import LabelSchema, PanopticTarget
from .transforms import PanopticTransform, build_targets


class PanopticDataset(Dataset[tuple[torch.Tensor, PanopticTarget, str]]):
    def __init__(
        self,
        manifest_path: str | Path,
        transform: PanopticTransform,
        schema: LabelSchema,
        max_samples: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        with self.manifest_path.open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        required = {"sample_id", "image_path", "semantic_path", "instance_path"}
        if self.rows and not required.issubset(self.rows[0]):
            raise ValueError(f"manifest must contain {sorted(required)}")
        if max_samples is not None:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError(f"manifest contains no samples: {self.manifest_path}")
        self.transform = transform
        self.schema = schema

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.manifest_path.parent / path).resolve()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, PanopticTarget, str]:
        row = self.rows[index]
        with Image.open(self._resolve(row["image_path"])) as source:
            image = source.convert("RGB")
        with Image.open(self._resolve(row["semantic_path"])) as source:
            semantic = Image.fromarray(np.asarray(source, dtype=np.uint8))
        with Image.open(self._resolve(row["instance_path"])) as source:
            instance = Image.fromarray(np.asarray(source, dtype=np.int32))
        image_tensor, semantic_tensor, instance_tensor = self.transform(image, semantic, instance)
        target = build_targets(
            semantic_tensor,
            instance_tensor,
            self.schema.thing_ids,
            center_sigma=self.transform.center_sigma,
            ignore_index=self.schema.ignore_index,
        )
        return image_tensor, target, row["sample_id"]


def panoptic_collate(
    batch: list[tuple[torch.Tensor, PanopticTarget, str]],
) -> tuple[torch.Tensor, list[PanopticTarget], list[str]]:
    images, targets, sample_ids = zip(*batch, strict=True)
    return torch.stack(list(images)), list(targets), list(sample_ids)
