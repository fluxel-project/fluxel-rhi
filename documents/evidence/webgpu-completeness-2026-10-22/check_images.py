"""Verify captured PNG integrity, dimensions and nonblank scene output (stdlib)."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import zlib

ROOT = Path(__file__).resolve().parent


def pixels(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset, compressed = 8, bytearray()
    while offset < len(data):
        size = struct.unpack_from(">I", data, offset)[0]
        kind, chunk = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + size]
        crc = struct.unpack_from(">I", data, offset + 8 + size)[0]
        assert zlib.crc32(kind + chunk) & 0xffffffff == crc
        if kind == b"IHDR":
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", chunk)
            assert depth == 8 and color in (2, 6) and compression == filtering == interlace == 0
        elif kind == b"IDAT":
            compressed.extend(chunk)
        offset += size + 12
    channels = 3 if color == 2 else 4
    raw = zlib.decompress(compressed)
    stride = width * channels
    assert len(raw) == (stride + 1) * height
    previous = bytearray(stride)
    histogram = Counter()
    for y in range(height):
        start = y * (stride + 1)
        mode, row = raw[start], bytearray(raw[start + 1:start + 1 + stride])
        for x in range(stride):
            a = row[x - channels] if x >= channels else 0
            b = previous[x]
            c = previous[x - channels] if x >= channels else 0
            if mode == 0:
                predictor = 0
            elif mode == 1:
                predictor = a
            elif mode == 2:
                predictor = b
            elif mode == 3:
                predictor = (a + b) // 2
            else:
                assert mode == 4
                p = a + b - c
                distances = [abs(p - a), abs(p - b), abs(p - c)]
                predictor = [a, b, c][distances.index(min(distances))]
            row[x] = (row[x] + predictor) & 255
        histogram.update(bytes(row[x:x + 3]) for x in range(0, stride, channels))
        previous = row
    return width, height, histogram


report = json.loads((ROOT / "scenes/report.json").read_text())
assert report["head"] == "907e67abb4cb39322973d47f7c6f8346bc9c666e"
assert report["source_status"] == "" and len(report["cases"]) == 15
results = []
for case in report["cases"]:
    path = ROOT / "scenes" / case["png"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case["png_sha256"]
    assert case["state"] == {"done": True, "error": None} and not case["diagnostics"]
    assert case["environment"]["isFallbackAdapter"] is False
    width, height, histogram = pixels(path)
    assert (width, height) == (1280, 720)
    foreground = width * height - histogram.most_common(1)[0][1]
    assert foreground > 10000 and len(histogram) > 16
    results.append({"example": case["example"], "pixels_outside_dominant_color": foreground,
                    "unique_rgb_colors": len(histogram), "result": "pass"})
print(json.dumps(results, indent=2))
