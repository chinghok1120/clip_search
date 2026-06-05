#!/usr/bin/env python3
"""
Text -> image semantic search on the PN (the query half of the system).

The image FAISS index was built with the TRT SigLIP2-vision engine. For search to be
valid, query text must be encoded into the SAME embedding space. SigLIP has NO separate
projection heads — the vision tower's pooler_output (what the engine outputs) and the
text tower's pooler_output (== SiglipModel.get_text_features) are the aligned contrastive
space. So we encode queries with the HF SigLIP text tower and dot-product against the index.

Reports per-query text-encode latency (<50 ms target) and the top-k matches.

Usage (PN venv):
  python search_query.py "a man in a red shirt" "person with a backpack" --topk 5
"""
import argparse
import json
import time

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, SiglipModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries", nargs="+")
    ap.add_argument("--index", default="embeddings/crowdhuman_siglip2-l-256-hf.faiss")
    ap.add_argument("--meta", default="embeddings/crowdhuman_siglip2-l-256-hf.json")
    ap.add_argument("--hf-model", default="google/siglip2-large-patch16-256")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=64)  # SigLIP canonical text length
    a = ap.parse_args()

    index = faiss.read_index(a.index)
    meta = json.load(open(a.meta))["images"]
    print(f"index: {index.ntotal} vectors, dim {index.d}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.hf_model)
    model = SiglipModel.from_pretrained(a.hf_model, torch_dtype=torch.float32).eval().cuda()

    def encode(texts):
        ids = tok(texts, padding="max_length", max_length=a.max_length,
                  truncation=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            e = model.get_text_features(**ids).float()
        e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy().astype(np.float32)

    # warmup (first call pays graph/alloc cost)
    encode(["warmup"])

    for q in a.queries:
        t0 = time.perf_counter()
        emb = encode([q])
        ms = (time.perf_counter() - t0) * 1000
        D, I = index.search(emb, a.topk)
        print(f"\nQUERY: {q!r}   (text-encode {ms:.1f} ms)")
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
            print(f"  {rank}. {score:+.4f}  {meta[idx]['filename']}")


if __name__ == "__main__":
    main()
