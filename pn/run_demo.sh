#!/usr/bin/env bash
#
# Bring up the PN CLIP-search demo. Run ON the PN, from the deploy root:
#
#   ./run_demo.sh                # build the index if missing, then serve on :8000
#   PORT=8001 ./run_demo.sh      # serve on a different port
#
# One-time prereqs (see README.md):
#   ./setup/setup_model.sh       # venv + deps + TRT engine (siglip2_l_256_hf_fp16.pth)
#   ./setup/setup_db.sh          # faiss-cpu
#   thumbnails at $IMAGES        # the image corpus (data — you provide)
#
# If embeddings/<NAME>.faiss is absent this encodes the corpus with the TRT engine and
# builds the index first; otherwise it goes straight to serving.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="$ROOT/venv/bin/python"
ENGINE="${ENGINE:-$ROOT/siglip2_l_256_hf_fp16.pth}"
IMAGES="${IMAGES:-$HOME/datasets/crowdhuman/train/images_960}"
NAME="${NAME:-crowdhuman_siglip2-l-256-hf}"
EMB="$ROOT/embeddings/$NAME"
PORT="${PORT:-8000}"

[ -x "$PY" ] || { echo "FATAL: venv missing ($PY). Run ./setup/setup_db.sh (and setup_model.sh) first."; exit 1; }

if [ ! -f "$EMB.faiss" ]; then
  echo ">> index $EMB.faiss missing — building it"
  [ -f "$ENGINE" ] || { echo "FATAL: TRT engine missing ($ENGINE). Run ./setup/setup_model.sh."; exit 1; }
  [ -d "$IMAGES" ] || { echo "FATAL: image corpus missing ($IMAGES). Provide thumbnails (see README.md)."; exit 1; }
  if [ ! -f "$EMB.npy" ]; then
    echo ">> encoding corpus -> $EMB.npy/.json"
    "$PY" "$ROOT/tools/encode_images_trt.py" --input "$IMAGES" --engine "$ENGINE" --out "$EMB"
  fi
  echo ">> building FAISS index -> $EMB.faiss"
  "$PY" "$ROOT/setup/build_faiss_index.py" --emb "$EMB.npy" --out "$EMB.faiss"
fi

echo ">> serving on http://0.0.0.0:$PORT  (Ctrl-C to stop)"
cd "$ROOT/web_pn"
exec "$PY" -m uvicorn pn_app:app --host 0.0.0.0 --port "$PORT"
