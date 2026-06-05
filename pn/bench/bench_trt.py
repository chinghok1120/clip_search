#!/usr/bin/env python3
"""Benchmark + validate a TensorRT engine for the EVA-02-L vision encoder.
Uses torch CUDA tensors as I/O buffers (no pycuda needed). TRT 10.x API.
"""
import argparse, time, statistics
import numpy as np
import torch
import tensorrt as trt

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="eva02_l_visual_fp16.engine")
ap.add_argument("--res", type=int, default=336)
ap.add_argument("--batches", default="1,8,16")
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--val-input", default="val_input.npy")
ap.add_argument("--val-ref", default="val_ref.npy")
args = ap.parse_args()

logger = trt.Logger(trt.Logger.WARNING)
with open(args.engine, "rb") as f, trt.Runtime(logger) as rt:
    engine = rt.deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()

# tensor names by I/O mode
in_name = out_name = None
for i in range(engine.num_io_tensors):
    n = engine.get_tensor_name(i)
    if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
        in_name = n
    else:
        out_name = n
in_dtype = trt.nptype(engine.get_tensor_dtype(in_name))
print(f"Engine: in='{in_name}' ({in_dtype.__name__}) out='{out_name}' | device {torch.cuda.get_device_name(0)}")

stream = torch.cuda.Stream()

def infer(inp):
    bs = inp.shape[0]
    ctx.set_input_shape(in_name, (bs, 3, args.res, args.res))
    out_shape = tuple(ctx.get_tensor_shape(out_name))
    out = torch.empty(out_shape, device="cuda", dtype=torch.float32)
    ctx.set_tensor_address(in_name, inp.data_ptr())
    ctx.set_tensor_address(out_name, out.data_ptr())
    ctx.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return out

# ---------- VALIDATION vs PyTorch reference (batch 1) ----------
x = np.load(args.val_input).astype(np.float32)          # (1,3,res,res)
ref = np.load(args.val_ref)                              # (1,768)
out = infer(torch.from_numpy(x).cuda()).cpu().numpy()
cos = float(np.sum(ref * out, -1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(out, axis=-1)))
print(f"VALIDATION  TRT vs PyTorch: cos_sim={cos:.6f} | max_abs_diff={np.abs(ref-out).max():.2e}  "
      f"-> {'OK ✓' if cos > 0.999 else ('CLOSE' if cos > 0.99 else 'DEGRADED ✗')}")
print("=" * 70)

# ---------- THROUGHPUT ----------
print(f"{'batch':>6} {'img/s':>10} {'ms/img':>9} {'img/min':>10}")
for bs in [int(b) for b in args.batches.split(",")]:
    inp = torch.randn(bs, 3, args.res, args.res, device="cuda", dtype=torch.float32)
    for _ in range(args.warmup):
        infer(inp)
    ts = []
    for _ in range(args.iters):
        s = time.time(); infer(inp); ts.append(time.time() - s)
    per_batch = statistics.median(ts)
    ips = bs / per_batch
    print(f"{bs:>6} {ips:>10.1f} {per_batch/bs*1000:>9.1f} {ips*60:>10.0f}")
