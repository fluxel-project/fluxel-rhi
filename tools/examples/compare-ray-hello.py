"""Compare equal-sized client captures of the official and RHI DXR samples."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--diff", type=Path)
    args = parser.parse_args()
    with Image.open(args.reference) as image:
        reference = image.convert("RGB")
    with Image.open(args.candidate) as image:
        candidate = image.convert("RGB")
    if reference.size != candidate.size:
        raise SystemExit(f"client extents differ: {reference.size} != {candidate.size}")
    difference = ImageChops.difference(reference, candidate)
    histogram = difference.histogram()
    red, green, blue = difference.split()
    max_channel = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    differing = reference.width * reference.height - max_channel.histogram()[0]
    report = {
        "reference": str(args.reference), "candidate": str(args.candidate),
        "extent": reference.size, "pixels": reference.width * reference.height,
        "different_pixels": differing,
        "max_channel_difference": max(i % 256 for i, count in enumerate(histogram) if count),
        "mean_channel_difference": ImageStat.Stat(difference).mean,
        "difference_bounds": difference.getbbox(),
    }
    if args.diff:
        args.diff.parent.mkdir(parents=True, exist_ok=True)
        difference.save(args.diff)
    print(json.dumps(report, indent=2))
    if differing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
