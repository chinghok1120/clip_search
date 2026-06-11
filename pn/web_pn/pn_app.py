#!/usr/bin/env python3
"""
PN semantic-search web demo (SigLIP2-L-256).

Query path only: the image FAISS index is prebuilt with the TRT vision engine; this app
encodes text queries with the HF SigLIP text tower (same embedding space) and serves a
thumbnail grid. Everything model-specific lives in PROFILE, so swapping 256 -> 384 (or any
model) is a one-dict change + its prebuilt index.

Run (PN venv):
  ./venv/bin/python -m uvicorn pn_app:app --host 0.0.0.0 --port 8000
"""
import json
import os
import time

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from transformers import AutoTokenizer, SiglipModel

# deploy root = parent of web_pn/ (this file's dir) — works whatever the folder is named
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- model profile (swap this dict + its index to change models) ---------------------
PROFILE = {
    "name": "SigLIP2-L/16-256",
    "hf_model": "google/siglip2-large-patch16-256",
    "index": os.path.join(ROOT, "embeddings/crowdhuman_siglip2-l-256-hf.faiss"),
    "meta": os.path.join(ROOT, "embeddings/crowdhuman_siglip2-l-256-hf.json"),
    "image_dir": os.path.expanduser("~/datasets/crowdhuman/train/images_960"),
    "max_length": 64,
}

app = FastAPI(title="PN CLIP Search")
STATE = {}


@app.on_event("startup")
def _load():
    STATE["index"] = faiss.read_index(PROFILE["index"])
    STATE["meta"] = json.load(open(PROFILE["meta"]))["images"]
    STATE["tok"] = AutoTokenizer.from_pretrained(PROFILE["hf_model"])
    STATE["model"] = SiglipModel.from_pretrained(
        PROFILE["hf_model"], torch_dtype=torch.float32).eval().cuda()
    # warmup
    _encode("warmup")
    print(f"[startup] {PROFILE['name']}: {STATE['index'].ntotal} vectors, dim {STATE['index'].d}")


def _encode(text: str) -> np.ndarray:
    ids = STATE["tok"]([text], padding="max_length", max_length=PROFILE["max_length"],
                       truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        e = STATE["model"].get_text_features(**ids).float()
    e = e / e.norm(dim=-1, keepdim=True)
    return e.cpu().numpy().astype(np.float32)


@app.get("/api/search")
def search(q: str, k: int = 24):
    if not q.strip():
        raise HTTPException(400, "empty query")
    t0 = time.perf_counter()
    emb = _encode(q)
    enc_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    D, I = STATE["index"].search(emb, k)
    search_ms = (time.perf_counter() - t1) * 1000
    results = [{"filename": STATE["meta"][int(i)]["filename"], "score": float(s)}
               for i, s in zip(I[0], D[0]) if i >= 0]
    return JSONResponse({"query": q, "encode_ms": round(enc_ms, 1),
                         "search_ms": round(search_ms, 2), "results": results})


@app.get("/thumb/{filename}")
def thumb(filename: str):
    # prevent path traversal; only serve plain filenames from the configured dir
    if "/" in filename or ".." in filename:
        raise HTTPException(400, "bad name")
    path = os.path.join(PROFILE["image_dir"], filename)
    if not os.path.isfile(path):
        raise HTTPException(404, "not found")
    return FileResponse(path)


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>PN CLIP Search</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 header{padding:16px 20px;background:#161a22;position:sticky;top:0}
 h1{font-size:16px;margin:0 0 10px} .sub{color:#8b93a7;font-size:12px;font-weight:400}
 form{display:flex;gap:8px} input{flex:1;padding:10px 12px;border-radius:8px;border:1px solid #2a3140;background:#0f1115;color:#e6e6e6;font-size:15px}
 button{padding:10px 18px;border:0;border-radius:8px;background:#3b82f6;color:#fff;font-size:15px;cursor:pointer}
 #meta{padding:8px 20px;color:#8b93a7;font-size:12px;min-height:18px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;padding:12px 20px}
 .card{background:#161a22;border-radius:8px;overflow:hidden} .card img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}
 .score{padding:5px 8px;font-size:12px;color:#9fb4d4}
</style></head><body>
<header><h1>PN CLIP Search <span class="sub">SigLIP2-L/16-256 · on the Jetson PN</span></h1>
<form id="f"><input id="q" placeholder="e.g. a person in a red jacket carrying a backpack" autofocus>
<button>Search</button></form></header>
<div id="meta"></div><div class="grid" id="g"></div>
<script>
const f=document.getElementById('f'),q=document.getElementById('q'),g=document.getElementById('g'),m=document.getElementById('meta');
f.onsubmit=async e=>{e.preventDefault();const query=q.value.trim();if(!query)return;
 m.textContent='searching…';g.innerHTML='';
 const r=await fetch('/api/search?q='+encodeURIComponent(query)+'&k=24');const d=await r.json();
 m.textContent=`${d.results.length} results · text-encode ${d.encode_ms} ms · vector-search ${d.search_ms} ms`;
 g.innerHTML=d.results.map(x=>`<div class="card"><img loading="lazy" src="/thumb/${x.filename}"><div class="score">${x.score.toFixed(3)} · ${x.filename}</div></div>`).join('');
};
</script></body></html>"""
