"""Decode the reference BC7 DDS for backends that accept RGBA textures.

Requires Pillow with DDS/BC7 support. The input is the unmodified NVIDIA asset.
"""
from pathlib import Path
from PIL import Image

assets = Path(__file__).resolve().parent.parent / "assets" / "19_ray_query_particles"
with Image.open(assets / "environment-map.dds") as image:
    image.convert("RGBA").save(assets / "environment-map-rgba.png")
