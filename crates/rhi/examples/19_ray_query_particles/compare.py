#!/usr/bin/env python3
"""Compare three RHI readbacks with the captured NVIDIA reference (requires Pillow)."""
import argparse
from pathlib import Path
from PIL import Image, ImageChops, ImageStat

parser = argparse.ArgumentParser()
parser.add_argument("reference", type=Path, help="directory containing frame-01.png through frame-03.png")
parser.add_argument("actual", type=Path, help="directory containing ray-query-particles-01.png through -03.png")
args = parser.parse_args()
regions = {"visible": (400, 0, 1000, 709), "smoke": (400, 300, 650, 650)}
sequences = [[], []]
for i in range(1, 4):
    pair = [Image.open(args.reference / f"frame-{i:02}.png").convert("RGB"),
            Image.open(args.actual / f"ray-query-particles-{i:02}.png").convert("RGB")]
    if any(image.size != (1000, 709) for image in pair):
        raise SystemExit("expected 1000x709 images")
    for name, box in regions.items():
        a, b = [image.crop(box) for image in pair]
        mae = sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3
        print(f"frame {i} {name}: RGB means {ImageStat.Stat(a).mean} / {ImageStat.Stat(b).mean}; MAE {mae:.2f}")
        if mae > 15:
            raise SystemExit(f"{name} exceeds visual error bound")
    for seq, image in zip(sequences, pair):
        seq.append(image.crop(regions["smoke"]))
for name, seq in zip(("reference", "actual"), sequences):
    motion = sum(ImageStat.Stat(ImageChops.difference(seq[0], seq[2])).mean) / 3
    print(f"{name} smoke frame 1 to 3 MAE: {motion:.2f}")
    if motion < 2:
        raise SystemExit(f"{name} smoke did not change enough")
