# ADR 0001: A Small, Traceable Panoptic Model

[简体中文](0001-readable-panoptic-baseline.zh-CN.md)

- Status: accepted
- Decision date: 2026-08-22

## Problem

A first panoptic project needs to show how labels become targets, how three model outputs become instances, and how a score is produced. A large framework would shorten the code but hide those steps. The project also needs to run on CPU and on a Kaggle GPU without making the decoder's memory use depend on an unbounded number of centers.

## Decision

Use a compact U-Net with three output heads:

- semantic logits for class prediction;
- a center heatmap for thing instances;
- two offset channels pointing to instance centers.

Keep semantic and instance masks separate in the input format. Apply the same geometry to image and masks, then create Gaussian centers and offsets on the transformed grid. During decoding, use same-class centers, a global center limit, and minimum region areas. Accumulate PQ by class over the complete split.

Version 1 does not represent crowd regions. Dataset-specific converters and evaluators must handle crowd and void rules outside this base format.

Store the final configuration, data fingerprint, environment, random state, and checkpoint state with each run. This makes a result reproducible without hiding the settings in a command history.

## Consequences

The code is small enough to trace with hand-built tensors and CPU tests. It is not the full Panoptic-DeepLab architecture and is not intended as a high-accuracy baseline. Adding another model or dataset should preserve the output shapes, data rules, checkpoint reconstruction, and metric tests. A change to those meanings requires a migration or a new schema version.
