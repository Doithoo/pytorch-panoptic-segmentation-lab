"""Convert an extracted Cityscapes tree to the project contract."""

from __future__ import annotations

import argparse

from panoptic_segmenter.data.cityscapes import convert_cityscapes_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="data/cityscapes")
    parser.add_argument("--symlink-images", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    args = parser.parse_args()
    output = convert_cityscapes_dataset(
        args.data_root,
        args.output_root,
        copy_images=not args.symlink_images,
        strict=not args.non_strict,
    )
    print(f"converted Cityscapes data to {output}")
