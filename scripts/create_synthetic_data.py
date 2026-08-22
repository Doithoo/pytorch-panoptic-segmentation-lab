"""Create synthetic panoptic data for smoke tests and tutorials."""

from __future__ import annotations

import argparse

from panoptic_segmenter.data.synthetic import create_synthetic_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    create_synthetic_dataset(args.output, args.count, args.size)
    print(f"created {args.count} samples in {args.output}")
