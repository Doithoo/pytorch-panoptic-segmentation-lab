"""Safe, atomic, versioned training checkpoints and run metadata."""

from __future__ import annotations

import hashlib
import os
import pickle
import platform
import random
import subprocess
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torchvision  # type: ignore[import-untyped]

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot be loaded or resumed safely."""


def save_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    serialized = dict(payload)
    serialized.setdefault("schema_version", CHECKPOINT_SCHEMA_VERSION)
    try:
        torch.save(serialized, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    source = Path(path)
    try:
        loaded = torch.load(source, map_location=map_location, weights_only=True)
    except (pickle.UnpicklingError, OSError, RuntimeError, EOFError) as exc:
        raise CheckpointCompatibilityError(f"cannot safely load checkpoint {source}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise CheckpointCompatibilityError("checkpoint must contain a mapping")
    if loaded.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(f"unsupported checkpoint schema_version {loaded.get('schema_version')!r}")
    required = {"config", "schema", "model_state", "epoch", "best_metric", "run_metadata"}
    missing = required - set(loaded)
    if missing:
        raise CheckpointCompatibilityError(f"checkpoint is missing fields: {sorted(missing)}")
    return loaded


def capture_rng_state() -> dict[str, object]:
    numpy_state = cast(tuple[str, Any, int, int, float], np.random.get_state(legacy=True))
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_state = state.get("torch")
    if isinstance(python_state, tuple):
        random.setstate(python_state)
    if isinstance(numpy_state, Mapping) and isinstance(numpy_state.get("keys"), torch.Tensor):
        np.random.set_state(
            (
                str(numpy_state["name"]),
                numpy_state["keys"].cpu().numpy().astype(np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.cpu())
    cuda_state = state.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, list) and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)


def build_run_metadata(device: torch.device, seed: int) -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "platform": platform.platform(),
        "device": str(device),
        "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": seed,
        "git_revision": _git_revision(),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
