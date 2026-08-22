"""Prepared-data integrity checks run before training."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .schema import LabelSchema


@dataclass(frozen=True)
class DataIssue:
    sample_id: str
    message: str


@dataclass(frozen=True)
class DataReport:
    split_counts: dict[str, int]
    inspected_samples: int
    issues: tuple[DataIssue, ...]

    def raise_for_issues(self) -> None:
        if self.issues:
            details = "\n".join(f"- {item.sample_id}: {item.message}" for item in self.issues[:20])
            remainder = len(self.issues) - 20
            suffix = f"\n- ... and {remainder} more" if remainder > 0 else ""
            raise ValueError(f"prepared-data preflight failed:\n{details}{suffix}")


def inspect_prepared_dataset(manifest_dir: str | Path, *, limit_per_split: int | None = None) -> DataReport:
    root = Path(manifest_dir).resolve()
    schema_path = root / "schema.yaml"
    metadata_path = root / "dataset.yaml"
    if not schema_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("prepared data requires schema.yaml and dataset.yaml")
    schema = LabelSchema.read_yaml(schema_path)
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    issues: list[DataIssue] = []
    if not isinstance(metadata, dict):
        raise ValueError("dataset.yaml must contain a mapping")
    schema_hash = _sha256(schema_path)
    if metadata.get("schema_sha256") != schema_hash:
        issues.append(DataIssue("dataset.yaml", "schema SHA-256 does not match schema.yaml"))
    declared_counts = metadata.get("split_counts", {})
    declared_hashes = metadata.get("manifest_sha256", {})
    counts: dict[str, int] = {}
    inspected = 0
    seen_ids: set[str] = set()
    for split in ("train", "valid", "test"):
        manifest = root / f"{split}.csv"
        if not manifest.is_file():
            issues.append(DataIssue(split, f"missing manifest {manifest.name}"))
            counts[split] = 0
            continue
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        counts[split] = len(rows)
        if not rows:
            issues.append(DataIssue(split, "split is empty"))
        if not isinstance(declared_hashes, dict) or declared_hashes.get(split) != _sha256(manifest):
            issues.append(DataIssue(split, "manifest SHA-256 does not match dataset.yaml"))
        if not isinstance(declared_counts, dict) or declared_counts.get(split) != len(rows):
            issues.append(DataIssue(split, "dataset.yaml split count does not match manifest"))
        for row in rows:
            sample_id = row.get("sample_id", "<missing>")
            if sample_id in seen_ids:
                issues.append(DataIssue(sample_id, "sample ID appears in more than one split"))
            seen_ids.add(sample_id)
        inspected_rows = rows if limit_per_split is None else rows[:limit_per_split]
        for row in inspected_rows:
            issues.extend(_inspect_row(manifest, row, schema))
            inspected += 1
    if isinstance(declared_hashes, dict) and all(name in declared_hashes for name in ("train", "valid", "test")):
        source = schema_hash + "".join(str(declared_hashes[name]) for name in ("train", "valid", "test"))
        identity = hashlib.sha256(source.encode()).hexdigest()
        if metadata.get("identity") != identity:
            issues.append(DataIssue("dataset.yaml", "prepared dataset identity is inconsistent"))
    return DataReport(counts, inspected, tuple(issues))


def _inspect_row(manifest: Path, row: dict[str, str], schema: LabelSchema) -> list[DataIssue]:
    sample_id = row.get("sample_id", "<missing>")
    required = ("image_path", "semantic_path", "instance_path")
    if any(not row.get(name) for name in required):
        return [DataIssue(sample_id, "manifest row is missing required paths")]
    paths = {name: _resolve(manifest, row[name]) for name in required}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        return [DataIssue(sample_id, "missing files: " + ", ".join(missing))]
    try:
        with Image.open(paths["image_path"]) as opened:
            image_size = opened.size
            opened.verify()
        with Image.open(paths["semantic_path"]) as opened:
            semantic_size = opened.size
            semantic = np.asarray(opened, dtype=np.int64).copy()
        with Image.open(paths["instance_path"]) as opened:
            instance_size = opened.size
            instance = np.asarray(opened, dtype=np.int64).copy()
    except (OSError, ValueError) as exc:
        return [DataIssue(sample_id, f"cannot decode sample: {exc}")]
    issues: list[DataIssue] = []
    if image_size != semantic_size or image_size != instance_size:
        issues.append(DataIssue(sample_id, "image, semantic, and instance dimensions differ"))
        return issues
    allowed = set(range(schema.num_classes)) | {schema.ignore_index}
    unexpected = sorted({int(value) for value in np.unique(semantic)} - allowed)
    if unexpected:
        issues.append(DataIssue(sample_id, f"semantic mask contains unsupported IDs {unexpected[:10]}"))
    if np.any(instance < 0):
        issues.append(DataIssue(sample_id, "instance mask contains negative IDs"))
    if np.any(instance[semantic == schema.ignore_index] != 0):
        issues.append(DataIssue(sample_id, "ignored pixels must use instance ID 0"))
    thing_mask = np.isin(semantic, schema.thing_ids)
    if np.any(thing_mask & (instance <= 0)):
        issues.append(DataIssue(sample_id, "thing pixels must have a positive instance ID"))
    if np.any((~thing_mask) & (semantic != schema.ignore_index) & (instance > 0)):
        issues.append(DataIssue(sample_id, "stuff pixels must use instance ID 0"))
    for instance_id in np.unique(instance[instance > 0]):
        class_ids = np.unique(semantic[instance == instance_id])
        if len(class_ids) != 1:
            issues.append(DataIssue(sample_id, f"instance {instance_id} spans multiple semantic classes"))
    return issues


def _resolve(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
