"""Create a contact sheet for prepared panoptic samples."""

from __future__ import annotations

import argparse

from panoptic_segmenter.data.preview import create_preview

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="prepared CSV, such as data/manifests/train.csv")
    parser.add_argument("--output", default="artifacts/dataset-preview.png")
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    print(create_preview(args.manifest, args.output, args.limit))
