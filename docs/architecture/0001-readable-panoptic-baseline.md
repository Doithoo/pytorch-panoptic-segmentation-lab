# ADR 0001: A Readable Panoptic Baseline

[简体中文](0001-readable-panoptic-baseline.zh-CN.md)

- Status: accepted
- Decision date: 2026-08-22

## Context

The project needs to teach semantic and instance reasoning without hiding the target contract behind a large framework. It also needs bounded evaluation and artifacts that can be audited on CPU and Kaggle.

## Decision

Use a compact U-Net with semantic, center, and offset heads. Store source truth as separate semantic and instance masks. Build Gaussian centers after synchronized geometry. Decode with same-class bounded center assignment. Accumulate PQ by class over a split. Keep crowd out of schema version 1 and state that limitation explicitly.

Use strict dataclass configuration, prepared-data identity, preflight, and versioned `weights_only=True` checkpoints. Make post-processing part of saved experiment semantics. Publish a deterministic synthetic Kaggle run before claiming any real benchmark.

## Consequences

The implementation is readable and can be tested with hand-built tensors. It is not architecturally equivalent to full Panoptic-DeepLab and cannot report official crowd-aware Cityscapes/COCO scores without adapters. A future model registry or crowd schema must preserve checkpoint reconstruction and metric tests, likely through a new explicit schema version.
