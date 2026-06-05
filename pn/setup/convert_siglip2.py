#!/usr/bin/env python3
"""
Convert an HF SigLIP2 vision tower to a serialized torch2trt FP16 engine for the PN.

Builds the engine, validates it against the fp32 PyTorch model (cosine), SERIALIZES
it to disk (torch2trt state_dict), then reloads and re-validates — so deployment
loads the engine instead of rebuilding (~minutes) every start.

The model is wrapped to return ONLY the pooled embedding -> the TRT engine has clean
single-tensor I/O (what the encode/search pipeline wants).

NOTE: this torch2trt path works for 256 (and smaller). At 384 torch2trt OOMs on the
16GB board — use the split export-ONNX -> trtexec path for that resolution instead.

Usage (on the PN, inside the venv):
  python convert_siglip2.py --hf-model google/siglip2-large-patch16-256 \
                            --out siglip2_l_256_hf_fp16.pth
"""
import argparse
import os
import sys
import time

import torch
import torch2trt
from torch2trt import TRTModule
from transformers import SiglipVisionModel


class VisionPooler(torch.nn.Module):
    """Wrap SiglipVisionModel to output only pooler_output (single tensor)."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).pooler_output


def cos(a, b):
    a = a.float(); b = b.float()
    an = a / a.norm(dim=-1, keepdim=True)
    bn = b / b.norm(dim=-1, keepdim=True)
    return float((an * bn).sum(-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-model", default="google/siglip2-large-patch16-256")
    ap.add_argument("--out", default="siglip2_l_256_hf_fp16.pth")
    ap.add_argument("--workspace-gb", type=int, default=3)
    a = ap.parse_args()

    print(f"loading {a.hf_model}", flush=True)
    base = SiglipVisionModel.from_pretrained(
        a.hf_model, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval().cuda()
    res = base.config.image_size
    embed = base.config.hidden_size
    print(f"image_size={res} embed_dim={embed}", flush=True)

    m = VisionPooler(base).eval().cuda()
    x = torch.randn(1, 3, res, res, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        ref = m(x)
    print(f"torch forward OK, embed {tuple(ref.shape)}", flush=True)

    print("converting via torch2trt (fp16, eager attn)...", flush=True)
    trt = torch2trt.torch2trt(
        m, [x], fp16_mode=True, use_onnx=True,
        max_workspace_size=(1024 ** 3) * a.workspace_gb,
    )
    with torch.no_grad():
        out = trt(x)
    print(f"[build]   cos_sim vs fp32 = {cos(ref, out):.6f}", flush=True)

    # Serialize (torch2trt stores the engine + I/O binding names in the state_dict)
    torch.save(trt.state_dict(), a.out)
    size_mb = os.path.getsize(a.out) / 1e6
    print(f"saved engine -> {a.out} ({size_mb:.0f} MB)", flush=True)

    # Reload into a bare TRTModule and re-validate (proves the serialized engine works)
    trt2 = TRTModule()
    trt2.load_state_dict(torch.load(a.out))
    with torch.no_grad():
        out2 = trt2(x)
    c2 = cos(ref, out2)
    print(f"[reloaded] cos_sim vs fp32 = {c2:.6f}", flush=True)
    # Correctness GATE: a degraded engine (version-drift / SDPA mis-compile) must fail loudly,
    # not ship silently. This is what lets the environment stay version-flexible — we verify the
    # OUTPUT rather than pinning every package.
    if c2 < 0.99:
        sys.exit(f"FATAL: serialized engine degraded (cos {c2:.4f} < 0.99) — not trustworthy. "
                 f"Likely a torch/torch2trt version mismatch; do NOT deploy this engine.")

    # Quick throughput (only meaningful under MAXN + jetson_clocks)
    with torch.no_grad():
        for _ in range(5):
            trt2(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            trt2(x)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / 20 * 1000
    print(f"throughput batch1: {ms:.1f} ms ({60000/ms:.0f}/min)  "
          f"[valid only under MAXN+jetson_clocks]", flush=True)
    print(f"peak GPU mem: {torch.cuda.max_memory_allocated()/1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
