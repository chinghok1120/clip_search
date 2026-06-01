#!/usr/bin/env python3
"""
Real-decode encode benchmark + index builder on the PN.

Software JPEG decode (PIL) -> SigLIP-256 preprocess -> serialized torch2trt FP16 engine.
Reports a decode / preprocess / encode time breakdown and end-to-end img/min — the
"real image generation time" the synthetic-tensor benchmark deliberately skipped.

Production note: a real PN would decode received JPEGs in HARDWARE (NVDEC/NVJPEG); the
decode column here is the SOFTWARE (CPU PIL) cost, an upper bound on that stage.

Saves embeddings as .npy + metadata .json. The FAISS/vector-DB index is built separately
once the DB choice is settled (so this step doesn't depend on it).

Usage (PN venv, MAXN + jetson_clocks for valid timing):
  python encode_images_trt.py --input ~/datasets/crowdhuman/train/images_960 \
        --engine siglip2_l_256_hf_fp16.pth --batch-size 8
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torch2trt import TRTModule

# SigLIP normalization: x in [-1, 1]  ->  mean 0.5, std 0.5
MEAN, STD = 0.5, 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--engine", default="siglip2_l_256_hf_fp16.pth")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--out", default="embeddings/crowdhuman_siglip2-l-256-hf")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.input, "*")))
    if a.limit:
        paths = paths[:a.limit]
    n = len(paths)
    print(f"{n} images | batch {a.batch_size} | res {a.res} | engine {a.engine}", flush=True)

    trt = TRTModule()
    trt.load_state_dict(torch.load(a.engine))
    with torch.no_grad():  # warmup
        trt(torch.zeros(1, 3, a.res, a.res, device="cuda"))
        torch.cuda.synchronize()

    t_decode = t_prep = t_enc = 0.0
    embs, meta = [], []
    res = a.res
    i = 0
    wall0 = time.perf_counter()
    while i < n:
        batch = paths[i:i + a.batch_size]
        i += a.batch_size
        tensors = []
        for p in batch:
            t0 = time.perf_counter()
            try:
                im = Image.open(p)
                im = im.convert("RGB")
                im.load()                       # force actual decode now
            except Exception:
                continue
            t1 = time.perf_counter()
            im = im.resize((res, res), Image.BICUBIC)
            arr = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))   # HWC -> CHW
            meta.append({"path": p, "filename": os.path.basename(p)})
            t2 = time.perf_counter()
            t_decode += t1 - t0
            t_prep += t2 - t1
        if not tensors:
            continue
        x = torch.stack(tensors).cuda().float()
        torch.cuda.synchronize(); t3 = time.perf_counter()
        with torch.no_grad():
            e = trt(x)
        torch.cuda.synchronize(); t_enc += time.perf_counter() - t3
        e = e.float()
        e = e / e.norm(dim=-1, keepdim=True)
        embs.append(e.cpu().numpy())
        if (i // a.batch_size) % 50 == 0:
            print(f"  {min(i, n)}/{n}", flush=True)
    wall = time.perf_counter() - wall0

    E = np.concatenate(embs, 0).astype(np.float32)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.save(a.out + ".npy", E)
    with open(a.out + ".json", "w") as f:
        json.dump({"num": len(meta), "dim": int(E.shape[1]),
                   "model": a.engine, "images": meta}, f)

    m = E.shape[0]
    compute = t_decode + t_prep + t_enc
    print("\n=== REAL-DECODE BENCHMARK (software decode) ===")
    print(f"images encoded: {m}   embed_dim: {E.shape[1]}")
    print(f"  decode  (PIL/CPU):       {t_decode:6.1f}s  {1000*t_decode/m:5.2f} ms/img  -> {60*m/t_decode:7.0f}/min")
    print(f"  preprocess (resize+norm):{t_prep:6.1f}s  {1000*t_prep/m:5.2f} ms/img  -> {60*m/t_prep:7.0f}/min")
    print(f"  encode  (TRT/GPU):       {t_enc:6.1f}s  {1000*t_enc/m:5.2f} ms/img  -> {60*m/t_enc:7.0f}/min")
    print(f"  -------------------------------------------")
    print(f"  END-TO-END (serial wall):{wall:6.1f}s  {1000*wall/m:5.2f} ms/img  -> {60*m/wall:7.0f}/min")
    print(f"  decode = {100*t_decode/compute:.0f}% of compute time "
          f"(this is what HW JPEG decode would offload)")
    print(f"  NOTE: serial wall is a lower bound; with parallel CPU decode workers, "
          f"throughput -> min(parallel_decode, encode={60*m/t_enc:.0f}/min)")
    print(f"saved: {a.out}.npy / {a.out}.json")


if __name__ == "__main__":
    main()
