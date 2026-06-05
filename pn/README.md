# PN deployment (`pn/`)

Everything that runs on the **Processing Node** (Jetson Orin Nano 16GB). This folder maps
**1:1 to `PN:~/clip_search/`** — deploy by syncing it there, then run the setup scripts to
build the artifact layer locally on the board.

```
pn/
├── run_demo.sh        build index (if missing) + serve the FastAPI demo
├── .deployignore      rsync excludes = the artifact layer (never synced)
├── setup/             one-time provisioning (deps + TRT engine + faiss)
│   ├── setup_model.sh     venv + transformers/torch2trt + builds the TRT FP16 engine
│   ├── convert_siglip2.py     (HERE-sibling of setup_model.sh — keep together)
│   ├── setup_db.sh        installs faiss-cpu
│   └── build_faiss_index.py   (HERE-sibling of setup_db.sh — keep together)
├── web_pn/pn_app.py   the query-path demo server (text → FAISS → thumbnail grid)
├── tools/             ops scripts: encode, resize, search, validate, 384 export, t2t
└── bench/             perf harnesses (jetson/trt/faiss-load) + dbbench/ DB head-to-head
```

## Code layer vs artifact layer

| Layer | What | Where it lives |
|---|---|---|
| **Code** (this folder, in git) | `*.sh`, `*.py` | synced from the repo |
| **Artifacts** (generated, gitignored) | `venv/`, `*.pth`/`*.engine`, `embeddings/`, bench `*.npy`/stores, `qdrant_bin/` | **built on the PN**, never synced — see `.deployignore` |

The setup scripts *recreate* the artifact layer, so you never copy multi-GB files around.

## Deploy to a fresh PN

Base requirement (NOT automated — `setup_model.sh` hard-fails otherwise): JetPack flashed
with the **NVIDIA Jetson torch wheel + TensorRT**, `git`, and ≥5 GB free in `$HOME`.

From the repo root on your dev machine:

```bash
./deploy.sh            # preview the sync (dry-run)
./deploy.sh --go       # rsync pn/ -> PN:~/clip_search/
```

Then **on the PN** (`cd ~/clip_search`):

| # | Step | Command |
|---|---|---|
| 1 | Build model engine | `./setup/setup_model.sh` |
| 2 | Install FAISS | `./setup/setup_db.sh` |
| 3 | Provide thumbnails *(data)* | put JPEGs at `~/datasets/crowdhuman/train/images_960` (or run `tools/resize_images.py` on the raw set) |
| 4 | Build index + serve | `./run_demo.sh` |

`run_demo.sh` encodes the corpus with the TRT engine → `embeddings/*.npy/.json`, builds the
FAISS index → `*.faiss`, then serves at **http://0.0.0.0:8000** (`PORT=…` to change).
Subsequent runs skip straight to serving once the index exists.

> The demo's **text** tower is the HF `SiglipModel` (loaded from the cache `setup_model.sh`
> populated); the **TRT engine** is used only to encode images into the index. Same
> embedding space, so queries and the index match.

## Model swap (256 → 384, or any model)

Change the single `PROFILE` dict in `web_pn/pn_app.py` and point it at a re-built index. The
main cost of a swap is re-encoding the corpus (re-indexing), not code.
