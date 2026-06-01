#!/usr/bin/env bash
#
# Repeatable vector-DB setup for the PN (faiss-cpu).
#
# faiss-cpu installs from a prebuilt aarch64 wheel on JetPack 6 (no source build).
# CPU-first by decision; whole index lives in RAM. IndexFlatIP at demo scale,
# IndexIVFPQ at production scale (pass --ivfpq to build_faiss_index.py).
#
# INSTALL TARGET (TARGET env), same semantics as setup_model.sh:
#   TARGET=venv  (default) -> isolated venv at $VENV
#   TARGET=user            -> ~/.local user scope (pip install --user), more compact
#
# Usage:
#   ./setup_db.sh                                        # install faiss-cpu only
#   ./setup_db.sh --build-index <emb.npy> <out.faiss>   # install + build the index
#   TARGET=user ./setup_db.sh ...                        # into ~/.local
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${TARGET:-venv}"

# resolve install target -> $PY / $PIP
if [ "$TARGET" = "user" ]; then
  echo ">> TARGET=user: installing into ~/.local (user scope), no venv"
  PY="python3"
  PIP="python3 -m pip install --user"
else
  VENV="${VENV:-$HOME/clip_search/venv}"
  if [ ! -x "$VENV/bin/python" ]; then
    echo ">> creating venv $VENV (--system-site-packages)"
    python3 -m venv --system-site-packages "$VENV"
  fi
  PY="$VENV/bin/python"
  PIP="$VENV/bin/pip install"
  echo ">> TARGET=venv: using $VENV"
fi

echo ">> installing faiss-cpu (numpy held <2 for torch 2.3) [TARGET=$TARGET]"
$PIP -q "numpy<2" faiss-cpu
"$PY" -c "import faiss; print('   faiss', faiss.__version__)"

if [ "${1:-}" = "--build-index" ]; then
  EMB="${2:?usage: --build-index <emb.npy> <out.faiss>}"
  OUT="${3:?usage: --build-index <emb.npy> <out.faiss>}"
  echo ">> building index from $EMB -> $OUT"
  "$PY" "$HERE/build_faiss_index.py" --emb "$EMB" --out "$OUT"
fi

echo ">> DB setup done."
