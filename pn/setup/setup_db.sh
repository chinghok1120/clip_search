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
#
# Light on resources (prebuilt wheel + RAM-resident IndexFlatIP at demo scale), so unlike
# setup_model.sh this has no memory guard — it just installs and, optionally, builds.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"      # deploy root (parent of setup/) — works whatever the folder is named
TARGET="${TARGET:-venv}"

# ---- pretty output helpers (same style as setup_model.sh) ----------------------------
step()   { echo; echo "── Step $1 ·············································· $2"; }
ok()     { echo "   OK  $*"; }
indent() { sed 's/^/   │ /'; }   # nest a sub-process's stdout/stderr under the current step

# does this run also build an index?
BUILD_INDEX=false
[ "${1:-}" = "--build-index" ] && BUILD_INDEX=true
$BUILD_INDEX && NSTEPS=3 || NSTEPS=2

# ---- banner --------------------------------------------------------------------------
echo "======================================================================"
echo "  PN vector-DB setup — faiss-cpu (CPU-first, RAM-resident index)"
echo "======================================================================"
echo "  target : $TARGET"
$BUILD_INDEX && echo "  index  : ${3:-?}  (from ${2:-?})"
echo "  $NSTEPS steps, ~1 min (prebuilt wheel — no compile)."

# ======================================================================================
step "1/$NSTEPS · python environment" "(~20 s first run, instant after)"
if [ "$TARGET" = "user" ]; then
  PY="python3"
  PIP="python3 -m pip install --user"
  ok "TARGET=user: installing into ~/.local (user scope), no venv"
else
  VENV="${VENV:-$ROOT/venv}"
  if [ ! -x "$VENV/bin/python" ]; then
    echo "   creating venv (--system-site-packages, inherits JetPack torch/tensorrt)"
    python3 -m venv --system-site-packages "$VENV"
  fi
  PY="$VENV/bin/python"
  PIP="$VENV/bin/pip install"
  ok "TARGET=venv: $VENV"
fi

# ======================================================================================
if "$PY" -c "import faiss" 2>/dev/null; then
  step "2/$NSTEPS · faiss-cpu" "(already cached — skipping)"
  ok "faiss already present: $("$PY" -c 'import faiss; print(faiss.__version__)')"
else
  step "2/$NSTEPS · install faiss-cpu" "(~20–40 s — prebuilt aarch64 wheel, no compile)"
  echo "   installing faiss-cpu (numpy held <2 for torch 2.3) ..."
  $PIP -q "numpy<2" faiss-cpu 2>&1 | indent
  ok "faiss $("$PY" -c 'import faiss; print(faiss.__version__)')"
fi

# ======================================================================================
if $BUILD_INDEX; then
  EMB="${2:?usage: --build-index <emb.npy> <out.faiss>}"
  IDX="${3:?usage: --build-index <emb.npy> <out.faiss>}"
  step "3/$NSTEPS · build FAISS index" "(seconds at demo scale; longer for large corpora)"
  echo "   $EMB  ->  $IDX"
  "$PY" "$HERE/build_faiss_index.py" --emb "$EMB" --out "$IDX" 2>&1 | indent
  RC=${PIPESTATUS[0]}
  [ "$RC" -eq 0 ] || { echo; echo "FATAL: index build failed (exit $RC). See the error above."; exit "$RC"; }
fi

# ======================================================================================
echo
echo "======================================================================"
echo "  DONE — vector DB ready"
$BUILD_INDEX && echo "    index: ${3}"
echo "  next:  ./run_demo.sh              # build index (if needed) + serve :8000"
echo "======================================================================"
