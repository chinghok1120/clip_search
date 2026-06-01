#!/usr/bin/env python3
"""
Validate a serialized torch2trt engine against the fp32 PyTorch model on REAL images.

Stronger than the single-synthetic-image check inside convert_siglip2.py: encodes N real
images through BOTH the fp32 SiglipVisionModel and the TRT fp16 engine and reports
  - per-image cosine vs fp32: mean / MIN (worst-case) / max
  - top-k nearest-neighbour ranking agreement (what actually affects search results)

Usage:
  python validate_engine.py                          # 200 CrowdHuman images, default engine
  python validate_engine.py --input /imgs --num 500  # other folder / count
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch2trt import TRTModule
from transformers import SiglipVisionModel

MEAN, STD = 0.5, 0.5


def load_batch(paths, res):
    ts = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((res, res), Image.BICUBIC)
        arr = (np.asarray(im, dtype=np.float32) / 255.0 - MEAN) / STD
        ts.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.expanduser("~/datasets/crowdhuman/train/images_960"),
                    help="image folder (default: resized CrowdHuman)")
    ap.add_argument("--num", type=int, default=200, help="number of images (default 200)")
    ap.add_argument("--engine", default="siglip2_l_256_hf_fp16.pth")
    ap.add_argument("--hf-model", default="google/siglip2-large-patch16-256")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=1, help="match the engine's batch profile")
    ap.add_argument("--min-cos", type=float, default=0.99, help="PASS threshold on worst-case cos")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.input, "*")))[:a.num]
    if not paths:
        sys.exit(f"no images found in {a.input}")
    n = len(paths)
    print(f"validating engine '{a.engine}' on {n} real images from {a.input}", flush=True)

    fp32_model = SiglipVisionModel.from_pretrained(
        a.hf_model, torch_dtype=torch.float32, attn_implementation="eager").eval().cuda()
    trt = TRTModule()
    trt.load_state_dict(torch.load(a.engine))

    R, O = [], []
    with torch.no_grad():
        for i in range(0, n, a.batch_size):
            x = load_batch(paths[i:i + a.batch_size], a.res).cuda().float()
            r = fp32_model(x).pooler_output.float()
            o = trt(x).float()
            R.append((r / r.norm(dim=-1, keepdim=True)).cpu())
            O.append((o / o.norm(dim=-1, keepdim=True)).cpu())
    R = torch.cat(R)
    O = torch.cat(O)

    # 1) matched-pair cosine: same image through fp32 vs fp16
    cos = (R * O).sum(-1)
    print(f"per-image cos vs fp32:  mean {cos.mean():.6f}   MIN {cos.min():.6f}   max {cos.max():.6f}",
          flush=True)

    # 2) ranking agreement: top-k neighbours within the set, fp32 vs fp16 (the search-relevant test)
    simR, simO = R @ R.T, O @ O.T
    eye = torch.eye(n, dtype=torch.bool)
    simR.masked_fill_(eye, -1.0)
    simO.masked_fill_(eye, -1.0)
    topR = simR.topk(a.topk, dim=-1).indices
    topO = simO.topk(a.topk, dim=-1).indices
    overlap = np.array([len(set(topR[i].tolist()) & set(topO[i].tolist())) / a.topk for i in range(n)])
    top1_agree = (topR[:, 0] == topO[:, 0]).float().mean().item()
    print(f"top-{a.topk} neighbour-set overlap (fp32 vs fp16): mean {100*overlap.mean():.1f}%   "
          f"min {100*overlap.min():.0f}%", flush=True)
    print(f"top-1 neighbour identical: {100*top1_agree:.1f}%", flush=True)

    ok = (cos.min().item() >= a.min_cos) and (overlap.mean() >= 0.90)
    print(("PASS" if ok else "WARN") +
          f": engine fidelity on real images (worst cos {cos.min():.5f} vs threshold {a.min_cos})",
          flush=True)
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
