#!/usr/bin/env python3
"""Benchmark a CLIP model on Jetson: image-encode throughput, text-encode latency, peak GPU mem.
Uses synthetic input tensors of the correct shape to isolate pure encode compute (no JPEG/disk noise).
"""
import argparse, time, statistics
import torch
import open_clip

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EVA02-L-14-336")
    ap.add_argument("--pretrained", default="merged2b_s6b_b61k")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--batches", default="1,8,16,32")
    ap.add_argument("--iters", type=int, default=20, help="timed iterations per batch size")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available!"
    dev = "cuda"
    use_fp16 = args.precision == "fp16"

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Loading {args.model} / {args.pretrained} (precision={args.precision})...")
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained)
    model = model.to(dev).eval()
    if use_fp16:
        model = model.half()
    tokenizer = open_clip.get_tokenizer(args.model)
    # infer input resolution from preprocess
    res = preprocess.transforms[0].size
    res = res if isinstance(res, int) else res[0]
    load_s = time.time() - t0
    dtype = torch.float16 if use_fp16 else torch.float32
    with torch.no_grad():
        embed_dim = model.encode_image(
            torch.randn(1, 3, res, res, device=dev, dtype=dtype)).shape[-1]
    print(f"Model loaded in {load_s:.1f}s | input {res}x{res} | embed_dim {embed_dim}")
    print("=" * 70)

    def sync(): torch.cuda.synchronize()

    # ---------- IMAGE ENCODE THROUGHPUT ----------
    print(f"{'batch':>6} {'img/s':>10} {'ms/img':>9} {'img/min':>10} {'peak MB':>9}")
    for bs in [int(x) for x in args.batches.split(",")]:
        x = torch.randn(bs, 3, res, res, device=dev, dtype=dtype)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model.encode_image(x)
            sync()
            ts = []
            for _ in range(args.iters):
                s = time.time(); _ = model.encode_image(x); sync()
                ts.append(time.time() - s)
        per_batch = statistics.median(ts)
        ips = bs / per_batch
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"{bs:>6} {ips:>10.1f} {per_batch/bs*1000:>9.1f} {ips*60:>10.0f} {peak_mb:>9.0f}")

    # ---------- TEXT ENCODE LATENCY ----------
    print("=" * 70)
    toks = tokenizer(["a woman in a blue dress walking through a crowd"]).to(dev)
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model.encode_text(toks)
        sync()
        ts = []
        for _ in range(args.iters):
            s = time.time(); _ = model.encode_text(toks); sync()
            ts.append(time.time() - s)
    print(f"Text encode (1 query): median {statistics.median(ts)*1000:.1f} ms | "
          f"p95 {sorted(ts)[int(0.95*len(ts))]*1000:.1f} ms")
    print(f"Total peak GPU mem this run: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")

if __name__ == "__main__":
    main()
