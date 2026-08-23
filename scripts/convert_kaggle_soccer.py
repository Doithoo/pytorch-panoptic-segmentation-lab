"""Convert the public Kaggle Soccer dataset to the project data contract."""

from __future__ import annotations

import argparse

from panoptic_segmenter.data.soccer import convert_kaggle_soccer_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="extracted Kaggle soccer dataset directory")
    parser.add_argument("--output", default="data/kaggle-soccer")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--resize-width", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()
    print(
        convert_kaggle_soccer_dataset(
            args.source,
            args.output,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
            resize_width=args.resize_width,
            jpeg_quality=args.jpeg_quality,
        )
    )
