"""Prepare portable manifests from a three-folder panoptic dataset."""

from __future__ import annotations

import csv
import hashlib
import os
import random
from pathlib import Path

import yaml

from .schema import LabelSchema


def prepare_paired_dataset(
    data_dir: str | Path,
    manifest_dir: str | Path,
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    schema: LabelSchema,
    group_file: str | Path | None = None,
) -> dict[str, Path]:
    root, output = Path(data_dir).resolve(), Path(manifest_dir).resolve()
    if len(ratios) != 3 or any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios must contain three non-negative values that sum to one")
    directories = {name: root / name for name in ("images", "semantic", "instance")}
    missing_directories = [name for name, path in directories.items() if not path.is_dir()]
    if missing_directories:
        raise FileNotFoundError("missing dataset directories: " + ", ".join(missing_directories))
    images = _indexed_files(directories["images"], {".jpg", ".jpeg", ".png"})
    semantic = _indexed_files(directories["semantic"], {".png"})
    instance = _indexed_files(directories["instance"], {".png"})
    key_sets = {"images": set(images), "semantic": set(semantic), "instance": set(instance)}
    if not all(key_sets.values()):
        raise ValueError("images, semantic, and instance directories must all contain supported files")
    common = set.intersection(*key_sets.values())
    if any(values != common for values in key_sets.values()):
        details = "; ".join(f"{name} unmatched={len(values - common)}" for name, values in key_sets.items())
        raise ValueError(f"dataset stems do not match exactly: {details}")
    ids = sorted(common)
    groups = _read_groups(group_file, ids) if group_file is not None else None
    splits = _split_ids(ids, ratios, seed, groups)
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for split, split_ids in splits.items():
        path = output / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_id", "image_path", "semantic_path", "instance_path"))
            writer.writeheader()
            for sample_id in split_ids:
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "image_path": os.path.relpath(images[sample_id], output),
                        "semantic_path": os.path.relpath(semantic[sample_id], output),
                        "instance_path": os.path.relpath(instance[sample_id], output),
                    }
                )
        result[split] = path
        hashes[split] = _sha256(path)
    schema_path = output / "schema.yaml"
    schema.write_yaml(schema_path)
    identity_source = _sha256(schema_path) + "".join(hashes[name] for name in ("train", "valid", "test"))
    metadata = {
        "format_version": 1,
        "data_dir": os.path.relpath(root, output),
        "seed": seed,
        "ratios": list(ratios),
        "split_counts": {name: len(values) for name, values in splits.items()},
        "manifest_sha256": hashes,
        "schema_sha256": _sha256(schema_path),
        "grouped_split": groups is not None,
        "group_file": os.path.relpath(Path(group_file).resolve(), output) if group_file is not None else None,
        "split_groups": _split_groups(splits, groups) if groups is not None else None,
        "identity": hashlib.sha256(identity_source.encode()).hexdigest(),
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    (output / "summary.txt").write_text(
        "\n".join([f"identity: {metadata['identity']}", *[f"{name}: {len(values)}" for name, values in splits.items()]])
        + "\n",
        encoding="utf-8",
    )
    return result


def _read_groups(path: str | Path | None, ids: list[str]) -> dict[str, str]:
    if path is None:
        return {}
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"group file not found: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"sample_id", "group_id"}.issubset(rows[0]):
        raise ValueError("group file must contain sample_id and group_id columns")
    groups: dict[str, str] = {}
    for row in rows:
        sample_id, group_id = row["sample_id"].strip(), row["group_id"].strip()
        if not sample_id or not group_id or sample_id in groups:
            raise ValueError("group file contains empty or duplicate sample/group IDs")
        groups[sample_id] = group_id
    if set(groups) != set(ids):
        raise ValueError("group file sample IDs must match dataset stems exactly")
    return groups


def _split_ids(
    ids: list[str], ratios: tuple[float, float, float], seed: int, groups: dict[str, str] | None
) -> dict[str, list[str]]:
    if groups is None:
        counts = _split_counts(len(ids), ratios)
        random.Random(seed).shuffle(ids)
        boundaries = (counts[0], counts[0] + counts[1])
        return {
            "train": ids[: boundaries[0]],
            "valid": ids[boundaries[0] : boundaries[1]],
            "test": ids[boundaries[1] :],
        }
    members: dict[str, list[str]] = {}
    for sample_id in ids:
        members.setdefault(groups[sample_id], []).append(sample_id)
    active = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if len(members) < len(active):
        raise ValueError("grouped split requires at least one group per non-empty split")
    rng = random.Random(seed)
    group_items = list(members.items())
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)
    targets = [len(ids) * ratio for ratio in ratios]
    split_ids: list[list[str]] = [[], [], []]
    for index, (_, sample_ids) in enumerate(group_items):
        if index < len(active):
            split_index = active[index]
        else:
            split_index = min(
                active, key=lambda item: (len(split_ids[item]) / max(targets[item], 1), len(split_ids[item]))
            )
        split_ids[split_index].extend(sample_ids)
    for values in split_ids:
        values.sort()
    return {"train": split_ids[0], "valid": split_ids[1], "test": split_ids[2]}


def _split_groups(splits: dict[str, list[str]], groups: dict[str, str] | None) -> dict[str, list[str]] | None:
    if groups is None:
        return None
    return {split: sorted({groups[sample_id] for sample_id in sample_ids}) for split, sample_ids in splits.items()}


def _indexed_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.stem in result:
                raise ValueError(f"duplicate sample stem {path.stem!r} in {directory}")
            result[path.stem] = path.resolve()
    return result


def _split_counts(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    required = sum(value > 0 for value in ratios)
    if total < required:
        raise ValueError(f"at least {required} samples are required for non-empty requested splits")
    raw = [total * value for value in ratios]
    counts = [int(value) for value in raw]
    for index in sorted(range(3), key=lambda item: raw[item] - counts[item], reverse=True)[: total - sum(counts)]:
        counts[index] += 1
    for index, ratio in enumerate(ratios):
        if ratio > 0 and counts[index] == 0:
            donor = max(range(3), key=lambda item: counts[item])
            counts[donor] -= 1
            counts[index] += 1
    return counts[0], counts[1], counts[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
