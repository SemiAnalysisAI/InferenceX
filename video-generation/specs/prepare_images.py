"""Rebuild a schema 3 spec's conditioning images from their sources and check each recorded hash.

Usage: prepare_images.py SPEC SOURCES OUTPUT

SOURCES holds each original as ``<commons_sha1>.<ext>``. OUTPUT receives ``<sha256>.png``, the
layout ``vgbench run --image-dir`` reads. Exits 1 if any image differs from the spec.
"""

import hashlib
import io
import struct
import sys
from pathlib import Path

import PIL
from PIL import Image

from video_spec import load_spec

PILLOW_VERSION = "11.3.0"
_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_chunks(data: bytes) -> list[bytes]:
    if not data.startswith(_SIGNATURE):
        raise ValueError("not a PNG file")
    chunks, offset = [], len(_SIGNATURE)
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        chunks.append(data[offset + 4 : offset + 8])
        offset += length + 12
    return chunks


def prepare(source: Path, box: list[int], size: tuple[int, int]) -> bytes:
    """Crop ``box``, LANCZOS-resize to ``size``, and encode 8-bit RGB PNG with no ancillary chunks."""
    with Image.open(source) as original:
        resized = original.convert("RGB").resize(size, Image.Resampling.LANCZOS, box=tuple(box))
    # A fresh image drops ICC, EXIF and DPI metadata that resize() copies from the source.
    clean = Image.frombytes("RGB", resized.size, resized.tobytes())
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG", compress_level=9)
    data = buffer.getvalue()
    header = struct.unpack(">IIBBBBB", data[16:29])
    if set(png_chunks(data)) != {b"IHDR", b"IDAT", b"IEND"} or header != (*size, 8, 2, 0, 0, 0):
        raise ValueError(f"{source.name}: PNG is not plain non-interlaced 8-bit RGB")
    return data


def main(argv: list[str]) -> int:
    if PIL.__version__ != PILLOW_VERSION:
        raise SystemExit(f"use Pillow {PILLOW_VERSION}; LANCZOS and zlib output vary by version")
    spec = load_spec(argv[1])
    sources, output = Path(argv[2]), Path(argv[3])
    output.mkdir(parents=True, exist_ok=True)
    size = (spec["generation"]["width"], spec["generation"]["height"])
    mismatches = 0
    for name, image in spec["images"].items():
        source = image["source"]
        [original] = sources.glob(source["commons_sha1"] + ".*")
        if hashlib.sha1(original.read_bytes()).hexdigest() != source["commons_sha1"]:
            raise SystemExit(f"{name}: {original} does not match its Commons SHA-1")
        data = prepare(original, source["crop_box"], size)
        digest = hashlib.sha256(data).hexdigest()
        (output / f"{digest}.png").write_bytes(data)
        matches = (digest, len(data)) == (image["sha256"], image["byte_size"])
        mismatches += not matches
        print(f"{'ok' if matches else 'DIFFERS'} {name} sha256={digest} byte_size={len(data)}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
