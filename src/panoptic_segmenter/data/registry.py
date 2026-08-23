"""Named dataset converter registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .cityscapes import convert_cityscapes_dataset
from .soccer import convert_kaggle_soccer_dataset

Converter = Callable[..., Path]
_CONVERTERS: dict[str, Converter] = {}


def register_converter(name: str, converter: Converter, *, overwrite: bool = False) -> None:
    """Register a converter that returns its prepared output directory."""
    if not name.strip():
        raise ValueError("converter name must be non-empty")
    if name in _CONVERTERS and not overwrite:
        raise ValueError(f"converter is already registered: {name}")
    _CONVERTERS[name] = converter


def available_converters() -> tuple[str, ...]:
    return tuple(sorted(_CONVERTERS))


def convert_dataset(name: str, data_root: str | Path, output_root: str | Path, **kwargs: Any) -> Path:
    try:
        converter = _CONVERTERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown dataset converter {name!r}; available: {available_converters()}") from exc
    output = converter(data_root, output_root, **kwargs)
    if not isinstance(output, Path):
        raise TypeError(f"converter {name!r} returned {type(output).__name__}, not pathlib.Path")
    return output


register_converter("cityscapes", convert_cityscapes_dataset)
register_converter("kaggle-soccer", convert_kaggle_soccer_dataset)
