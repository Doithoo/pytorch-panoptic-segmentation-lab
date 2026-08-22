"""Canonical labels used by semantic and instance branches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ClassDefinition:
    id: int
    name: str
    isthing: bool
    color: tuple[int, int, int]


@dataclass(frozen=True)
class LabelSchema:
    classes: tuple[ClassDefinition, ...]
    ignore_index: int = 255

    def __post_init__(self) -> None:
        if not self.classes or [item.id for item in self.classes] != list(range(len(self.classes))):
            raise ValueError("class ids must be contiguous from zero")
        if any(not item.name.strip() for item in self.classes):
            raise ValueError("class names must be non-empty")
        if len({item.name for item in self.classes}) != len(self.classes):
            raise ValueError("class names must be unique")
        if any(
            len(item.color) != 3 or any(channel < 0 or channel > 255 for channel in item.color) for item in self.classes
        ):
            raise ValueError("class colors must contain three values between zero and 255")
        if self.ignore_index < 0 or any(item.id == self.ignore_index for item in self.classes):
            raise ValueError("ignore_index must be non-negative and not a class id")

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def thing_ids(self) -> tuple[int, ...]:
        return tuple(item.id for item in self.classes if item.isthing)

    @property
    def stuff_ids(self) -> tuple[int, ...]:
        return tuple(item.id for item in self.classes if not item.isthing)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ignore_index": self.ignore_index,
            "classes": [
                {"id": item.id, "name": item.name, "isthing": item.isthing, "color": list(item.color)}
                for item in self.classes
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LabelSchema:
        classes = []
        for item in raw.get("classes", []):
            if not isinstance(item, dict):
                raise ValueError("each class definition must be a mapping")
            raw_color = item["color"]
            if not isinstance(raw_color, (list, tuple)) or len(raw_color) != 3:
                raise ValueError("class color must contain exactly three channels")
            if not isinstance(item.get("isthing"), bool):
                raise ValueError("class isthing must be a boolean")
            color = (int(raw_color[0]), int(raw_color[1]), int(raw_color[2]))
            classes.append(
                ClassDefinition(
                    id=int(item["id"]),
                    name=str(item["name"]),
                    isthing=bool(item["isthing"]),
                    color=color,
                )
            )
        return cls(tuple(classes), int(raw.get("ignore_index", 255)))

    @classmethod
    def read_yaml(cls, path: str | Path) -> LabelSchema:
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))

    def write_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


PanopticTarget = dict[str, Any]


def default_label_schema() -> LabelSchema:
    """Return the package's synthetic learning schema without repository files."""
    return LabelSchema(
        classes=(
            ClassDefinition(0, "background", False, (32, 32, 32)),
            ClassDefinition(1, "person", True, (230, 80, 80)),
            ClassDefinition(2, "road", False, (80, 120, 180)),
        ),
        ignore_index=255,
    )
