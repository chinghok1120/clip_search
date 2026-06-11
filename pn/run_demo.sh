#!/usr/bin/env bash
#
# Bring up the PN CLIP-search demo. Run ON the PN, from the deploy root.
# Encodes a thumbnail corpus into a FAISS index (once), then serves a text->image
# search UI on :8000. Re-runs skip straight to serving once the index exists.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# ---- defaults (each overridable by env or flag) --------------------------------------
IMAGES="${IMAGES:-$HOME/datasets/crowdhuman/train/images_960}"
PORT="${PORT:-8000}"
NAME="${NAME:-crowdhuman_siglip2-l-256-hf}"
ENGINE="${ENGINE:-$ROOT/siglip2_l_256_hf_fp16.pth}"

show_help() {
  cat <<EOF
Usage: ./run_demo.sh [OPTIONS] [IMAGE_DIR]

Encode a folder of thumbnail JPEGs into a FAISS index (one-time), then serve the
PN semantic-search UI (type a query -> matching thumbnails) on a web port.

ARGUMENTS:
  IMAGE_DIR          Folder of thumbnail JPEGs to index and display.
                     Default: ~/datasets/crowdhuman/train/images_960

OPTIONS:
  --images DIR       Same as the positional IMAGE_DIR
  --port N           Port to serve on (default: $PORT)
  --name NAME        Index basename under embeddings/ (default: $NAME)
  --engine FILE      TRT engine path (default: <root>/siglip2_l_256_hf_fp16.pth)
  -h, --help         Show this help

PREREQUISITES (one-time — see README.md):
  ./setup/setup_model.sh     # venv + TRT engine
  ./setup/setup_db.sh        # faiss-cpu
  thumbnails at IMAGE_DIR     # the corpus you want to search (you provide)

WHAT IT DOES:
  1. If embeddings/<NAME>.faiss is missing: encode IMAGE_DIR with the TRT engine,
     then build the FAISS index. (One-time per corpus; minutes for large sets.)
  2. Serve the UI at http://0.0.0.0:<port>  (Ctrl-C to stop).
  IMAGE_DIR is still needed on re-runs — the UI serves the matching thumbnails from it.

EXAMPLES:
  ./run_demo.sh                                # default dataset, port $PORT
  ./run_demo.sh ~/datasets/my_thumbs           # custom image dir (positional)
  ./run_demo.sh --images ~/data/cam1 --port 8080
  IMAGES=~/data/cam1 ./run_demo.sh             # via env

NO DATASET YET?
  Any folder of JPEGs works. To downscale a raw set into 960px thumbnails:
    ./tools/resize_images.py --input <raw_dir> --output $IMAGES
EOF
}

# ---- parse args (flags + one positional = IMAGE_DIR) ---------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) show_help; exit 0 ;;
    --images)  IMAGES="${2:?--images needs a path}"; shift 2 ;;
    --port)    PORT="${2:?--port needs a number}"; shift 2 ;;
    --name)    NAME="${2:?--name needs a value}"; shift 2 ;;
    --engine)  ENGINE="${2:?--engine needs a path}"; shift 2 ;;
    -*)        echo "unknown option: $1  (try --help)"; exit 1 ;;
    *)         IMAGES="$1"; shift ;;
  esac
done

PY="$ROOT/venv/bin/python"
EMB="$ROOT/embeddings/$NAME"

# ---- helpers (same style as setup_*.sh) ----------------------------------------------
phase()  { echo; echo "── $1 ·············································· $2"; }
ok()     { echo "   OK  $*"; }
indent() { sed 's/^/   │ /'; }

echo "======================================================================"
echo "  PN CLIP-search demo"
echo "======================================================================"
echo "  images : $IMAGES"
echo "  index  : $EMB.faiss"
echo "  serve  : http://0.0.0.0:$PORT"

# ======================================================================================
phase "preflight" "(~2 s)"
[ -x "$PY" ] || { echo "FATAL: venv missing ($PY). Run ./setup/setup_model.sh and ./setup/setup_db.sh first."; exit 1; }
ok "venv present"

# image dir is required to build the index AND to serve thumbnails. Validate it; if it's
# missing, explain how to provide one and (when interactive) prompt for a path.
while [ ! -d "$IMAGES" ]; do
  echo
  echo "!! image directory not found: $IMAGES"
  echo "   The demo needs a folder of thumbnail JPEGs to index and to display results."
  echo "   Provide one by:"
  echo "     • passing it:   ./run_demo.sh /path/to/jpegs"
  echo "     • env var:      IMAGES=/path/to/jpegs ./run_demo.sh"
  echo "     • making some:  ./tools/resize_images.py --input <raw_dir> --output $IMAGES"
  if [ -t 0 ]; then
    read -e -r -p "   Enter image directory now (blank to abort): " ans || exit 1
    [ -n "$ans" ] || { echo "   aborted."; exit 1; }
    IMAGES="${ans/#\~/$HOME}"          # expand a leading ~
  else
    echo "   (non-interactive shell — aborting; pass a path or set IMAGES=)"
    exit 1
  fi
done
ok "images: $IMAGES"

# ======================================================================================
if [ ! -f "$EMB.faiss" ]; then
  [ -f "$ENGINE" ] || { echo "FATAL: TRT engine missing ($ENGINE). Run ./setup/setup_model.sh."; exit 1; }
  NIMG=$(find "$IMAGES" -maxdepth 1 -type f | wc -l)

  if [ ! -f "$EMB.npy" ]; then
    phase "encode corpus" "(minutes — $NIMG images, ~thousands/min)"
    echo "   $IMAGES  ->  $EMB.npy/.json"
    "$PY" "$ROOT/tools/encode_images_trt.py" --input "$IMAGES" --engine "$ENGINE" --out "$EMB" 2>&1 | indent
    RC=${PIPESTATUS[0]}; [ "$RC" -eq 0 ] || { echo; echo "FATAL: encoding failed (exit $RC)."; exit "$RC"; }
  fi

  phase "build FAISS index" "(seconds at demo scale)"
  echo "   $EMB.npy  ->  $EMB.faiss"
  "$PY" "$ROOT/setup/build_faiss_index.py" --emb "$EMB.npy" --out "$EMB.faiss" 2>&1 | indent
  RC=${PIPESTATUS[0]}; [ "$RC" -eq 0 ] || { echo; echo "FATAL: index build failed (exit $RC)."; exit "$RC"; }
else
  ok "index exists — skipping encode/build"
fi

# ======================================================================================
phase "serve" "(Ctrl-C to stop)"
echo "   open  http://0.0.0.0:$PORT  in a browser on the same network"
# Tell the server which index + images this run chose (pn_app honors these env overrides).
export INDEX="$EMB.faiss" META="$EMB.json" IMAGE_DIR="$IMAGES"
cd "$ROOT/web_pn"
exec "$PY" -m uvicorn pn_app:app --host 0.0.0.0 --port "$PORT"
