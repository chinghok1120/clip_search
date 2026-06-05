#!/usr/bin/env python3
"""
Resize a directory of images to fit within a max box, keeping aspect ratio,
DOWNSCALE-ONLY. Images already within the box are copied unchanged (no re-encode,
so their original quality/decode cost is preserved).

Purpose: the PN (Jetson) will receive ~640x360 camera thumbnails in production.
CrowdHuman frames are 1920x1080, so decoding them on the PN over-states JPEG decode
cost. Resizing to ~960x540 makes the decode benchmark representative.

Runs CPU-only (Pillow + multiprocessing). Safe to run inside a venv without
touching the GPU.

Usage:
  python resize_images.py --input  /path/to/images \
                          --output /path/to/images_960 \
                          --max-width 960 --max-height 540 --quality 85 --workers 8
"""
import argparse
import os
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process(args):
    src, dst, max_w, max_h, quality = args
    try:
        with Image.open(src) as im:
            w, h = im.size
            # Downscale-only: if already within the box, copy as-is (no re-encode).
            if w <= max_w and h <= max_h:
                shutil.copy2(src, dst)
                return ("copied", os.path.getsize(src), os.path.getsize(dst))
            im = im.convert("RGB")
            im.thumbnail((max_w, max_h), Image.LANCZOS)  # keeps aspect, never upscales
            im.save(dst, "JPEG", quality=quality)
        return ("resized", os.path.getsize(src), os.path.getsize(dst))
    except Exception as e:
        return ("failed:" + type(e).__name__, 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-width", type=int, default=960)
    ap.add_argument("--max-height", type=int, default=540)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    in_dir, out_dir = Path(a.input), Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in in_dir.iterdir() if p.suffix.lower() in EXTS]
    print(f"Found {len(files)} images in {in_dir}")
    print(f"Target box {a.max_width}x{a.max_height}, quality {a.quality}, "
          f"{a.workers} workers (downscale-only; smaller images copied as-is)")

    # Resized outputs are always .jpg; copies keep their original extension.
    jobs = []
    for p in files:
        # decide extension lazily inside worker? simpler: name by stem + .jpg for resized,
        # but we don't know size yet. Use .jpg for all resized; copy keeps ext.
        out = out_dir / (p.stem + ".jpg")
        jobs.append((str(p), str(out), a.max_width, a.max_height, a.quality))

    t0 = time.time()
    with Pool(a.workers) as pool:
        results = pool.map(process, jobs, chunksize=16)
    dt = time.time() - t0

    n_resized = sum(1 for r in results if r[0] == "resized")
    n_copied = sum(1 for r in results if r[0] == "copied")
    n_failed = sum(1 for r in results if r[0].startswith("failed"))
    in_bytes = sum(r[1] for r in results)
    out_bytes = sum(r[2] for r in results)

    print(f"\nDone in {dt:.1f}s")
    print(f"  resized: {n_resized}   copied (already small): {n_copied}   failed: {n_failed}")
    print(f"  input:  {in_bytes/1e9:.2f} GB")
    print(f"  output: {out_bytes/1e9:.2f} GB  ({out_bytes/max(in_bytes,1)*100:.0f}% of input)")
    print(f"  output dir: {out_dir}")


if __name__ == "__main__":
    main()
