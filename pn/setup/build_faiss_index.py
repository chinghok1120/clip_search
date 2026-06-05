#!/usr/bin/env python3
"""
Build a faiss-cpu index from normalized embeddings (.npy).

IndexFlatIP on L2-normalized vectors == cosine similarity, exact search — the right
choice at demo scale (whole index in RAM, sub-ms search). For production scale
(~tens of millions of vectors that won't fit flat in 16 GB), use --ivfpq to build a
compressed IndexIVFPQ instead.

Usage:
  python build_faiss_index.py --emb embeddings/crowdhuman_siglip2-l-256-hf.npy \
                              --out embeddings/crowdhuman_siglip2-l-256-hf.faiss
"""
import argparse
import time

import faiss
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help=".npy embeddings (N x D), L2-normalized")
    ap.add_argument("--out", required=True, help="output .faiss path")
    ap.add_argument("--renorm", action="store_true", help="re-normalize rows before indexing")
    ap.add_argument("--ivfpq", action="store_true", help="build IndexIVFPQ (production scale) instead of flat")
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--m", type=int, default=64, help="PQ subquantizers (bytes/vector)")
    a = ap.parse_args()

    E = np.load(a.emb).astype(np.float32)
    if a.renorm:
        faiss.normalize_L2(E)
    n, d = E.shape
    print(f"{n} vectors, dim {d}", flush=True)

    if a.ivfpq:
        quant = faiss.IndexFlatIP(d)
        idx = faiss.IndexIVFPQ(quant, d, a.nlist, a.m, 8, faiss.METRIC_INNER_PRODUCT)
        t = time.perf_counter()
        idx.train(E)
        print(f"train: {time.perf_counter()-t:.1f}s", flush=True)
        idx.add(E)
        idx.nprobe = 16
    else:
        idx = faiss.IndexFlatIP(d)
        t = time.perf_counter()
        idx.add(E)
        print(f"add: {1000*(time.perf_counter()-t):.1f} ms", flush=True)

    faiss.write_index(idx, a.out)
    print(f"wrote {a.out}  ntotal={idx.ntotal}", flush=True)

    # Self-search sanity: top-1 of each vector should be itself (cos=1.0)
    q = E[:200]
    t = time.perf_counter()
    D, I = idx.search(q, 5)
    dt = 1000 * (time.perf_counter() - t)
    top1_self = float((I[:, 0] == np.arange(200)).mean())
    print(f"search 200x top-5: {dt:.2f} ms total, {dt/200:.4f} ms/query", flush=True)
    print(f"top-1 == self: {100*top1_self:.0f}%   (top-1 sim range {D[:,0].min():.3f}..{D[:,0].max():.3f})", flush=True)


if __name__ == "__main__":
    main()
