"""Panoptic target schema and dataset metadata."""

from .cityscapes import (
    cityscapes_categories,
    cityscapes_panoptic_ids,
    cityscapes_schema,
    cityscapes_segments_info,
    convert_cityscapes_dataset,
    convert_cityscapes_labels,
    discover_cityscapes_root,
    read_cityscapes_panoptic_png,
    write_cityscapes_panoptic_json,
    write_cityscapes_panoptic_png,
)
from .dataset import PanopticDataset, panoptic_collate
from .inspection import DataIssue, DataReport, inspect_prepared_dataset
from .registry import available_converters, convert_dataset, register_converter
from .schema import ClassDefinition, LabelSchema, PanopticTarget, default_label_schema
from .soccer import convert_kaggle_soccer_dataset, rasterize_soccer_annotations, soccer_schema
from .synthetic import create_synthetic_dataset

__all__ = [
    "ClassDefinition",
    "DataIssue",
    "DataReport",
    "LabelSchema",
    "PanopticTarget",
    "PanopticDataset",
    "default_label_schema",
    "create_synthetic_dataset",
    "cityscapes_schema",
    "convert_cityscapes_dataset",
    "convert_cityscapes_labels",
    "discover_cityscapes_root",
    "cityscapes_categories",
    "cityscapes_panoptic_ids",
    "cityscapes_segments_info",
    "read_cityscapes_panoptic_png",
    "write_cityscapes_panoptic_json",
    "write_cityscapes_panoptic_png",
    "inspect_prepared_dataset",
    "available_converters",
    "convert_dataset",
    "register_converter",
    "soccer_schema",
    "convert_kaggle_soccer_dataset",
    "rasterize_soccer_annotations",
    "panoptic_collate",
]
