#!/usr/bin/env bash
#
# Repeatable CLIP-model setup for the PN: installs the conversion layer and builds +
# serializes the SigLIP2-256 HF -> torch2trt FP16 engine.
#
# Assumes the JetPack base is already present (a flashed Jetson has these):
#   - torch (NVIDIA Jetson CUDA build)   - tensorrt (JetPack)
# This script NEVER installs/changes torch or tensorrt (a pip torch would be CPU-only
# and lose the GPU). A post-install guard asserts they were left intact.
#
# INSTALL TARGET (TARGET env):
#   TARGET=venv  (default) -> isolated venv at $VENV (--system-site-packages inherits
#                             the JetPack torch/tensorrt). Safe; cleanly removable.
#   TARGET=user            -> ~/.local user scope (pip install --user). More compact,
#                             no venv folder. Shares ~/.local with other superrx Python
#                             work, so prefer this only after validating in venv.
#
# Usage:
#   ./setup_model.sh                       # venv (default)
#   TARGET=user ./setup_model.sh           # into ~/.local
#   HF_MODEL=... OUT=... ./setup_model.sh  # override model / output path
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"      # deploy root (parent of setup/) — works whatever the folder is named
HF_MODEL="${HF_MODEL:-google/siglip2-large-patch16-256}"
OUT="${OUT:-$ROOT/siglip2_l_256_hf_fp16.pth}"
TARGET="${TARGET:-venv}"

# Memory the engine build actually needs at peak (measured on Orin Nano 16GB, SigLIP2-L/256):
#   ~10.2 GB CPU (process RSS) + ~2.3 GB GPU = ~12.5 GB on the shared unified pool.
NEED_GB=13                     # ~12.5 rounded up; the hard floor / warn band derive from this

# ---- pretty output helpers -----------------------------------------------------------
step()   { echo; echo "── Step $1 ·············································· $2"; }
ok()     { echo "   OK  $*"; }
warn()   { echo "   !!  $*"; }
indent() { sed 's/^/   │ /'; }   # nest a sub-process's stdout/stderr under the current step
mem_mitigations() {
  cat <<'EOF'
   ── how to free memory for the build ──────────────────────────────────
     • stop heavy services :  sudo systemctl stop docker  ;  killall <app>
     • run headless (no GUI):  sudo systemctl isolate multi-user.target
     • add swap (needs sudo):  sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile \
                               && sudo mkswap /swapfile && sudo swapon /swapfile
     • lower TRT scratch    :  convert_siglip2.py --workspace-gb 2   (default 3)
     • reboot to clear leaks/zram, then re-run this script
   ──────────────────────────────────────────────────────────────────────
EOF
}

# ---- banner --------------------------------------------------------------------------
echo "======================================================================"
echo "  PN model setup — ${HF_MODEL##*/}  ->  TensorRT FP16 engine"
echo "======================================================================"
echo "  target : $TARGET"
echo "  output : $OUT"
echo "  4 steps, ~5–9 min on first run (deps compile + 1.2GB model download);"
echo "           ~2 min on re-runs once deps are cached."

# ======================================================================================
step "1/4 · preflight checks" "(~5 s)"
# Philosophy: require CAPABILITIES, treat VERSIONS as advisory. Correctness is gated AFTER
# the build by convert_siglip2.py's cos>=0.99 check — so this stays portable to newer PNs.

# git (torch2trt installs from git)
command -v git >/dev/null 2>&1 || { echo "FATAL: 'git' is required (torch2trt installs from git)."; exit 1; }

# disk: deps + ~1.2GB model + 940MB engine
FREE_GB=$(df -Pk "$HOME" | awk 'NR==2{print int($4/1048576)}')
if [ "${FREE_GB:-0}" -lt 5 ]; then
  echo "FATAL: only ${FREE_GB}GB free in \$HOME; need >=5GB (deps + ~1.2GB model + 940MB engine)."; exit 1
fi
ok "disk: ${FREE_GB}GB free (need >=5GB)"

# memory: the build peaks ~12.5GB on the unified pool. Count RAM + free swap (the build
# tolerates spilling to swap — just slower). Hard-fail if it can't fit even with swap;
# warn if it only fits by dipping into swap.
AVAIL_KB=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
SWAPFREE_KB=$(awk '/^SwapFree:/{print $2}' /proc/meminfo)
AVAIL_GB=$(( AVAIL_KB / 1048576 ))
SWAP_GB=$(( SWAPFREE_KB / 1048576 ))
TOTAL_GB=$(( (AVAIL_KB + SWAPFREE_KB) / 1048576 ))
if [ "$TOTAL_GB" -lt $((NEED_GB - 1)) ]; then
  echo "FATAL: only ${TOTAL_GB}GB usable (${AVAIL_GB}GB RAM + ${SWAP_GB}GB swap); the engine build"
  echo "       peaks ~12.5GB and WILL be OOM-killed (you'd lose ~2 min building, then 'Killed')."
  mem_mitigations
  exit 1
elif [ "$AVAIL_GB" -lt "$NEED_GB" ]; then
  warn "memory: only ${AVAIL_GB}GB RAM free (+${SWAP_GB}GB swap) — build peaks ~12.5GB, so it"
  warn "        will spill into swap and run slower. To stay in RAM, free ~$((NEED_GB - AVAIL_GB))GB:"
  mem_mitigations
else
  ok "memory: ${AVAIL_GB}GB RAM free (+${SWAP_GB}GB swap) — build peaks ~12.5GB"
fi

# capabilities (checked against the JetPack system python — fail before we build a venv)
python3 - <<'PY'
import sys
try:
    import torch
except Exception as e:
    sys.exit(f"FATAL: torch not importable from the JetPack base ({type(e).__name__}: {e}). "
             f"Provision JetPack + the NVIDIA Jetson torch wheel first.")
if not torch.cuda.is_available():
    sys.exit("FATAL: torch present but CUDA unavailable — cannot build a GPU engine.")
try:
    import tensorrt
except Exception as e:
    sys.exit(f"FATAL: tensorrt not importable ({type(e).__name__}). Install the JetPack TensorRT package.")
if not torch.__version__.startswith("2.3"):
    print(f"   !!  torch {torch.__version__} != validated 2.3.x — proceeding; if the pinned "
          f"torch2trt commit fails to export, a torch-matched commit may be needed.")
if not str(tensorrt.__version__).startswith("10.3"):
    print(f"   !!  tensorrt {tensorrt.__version__} != validated 10.3.x — proceeding.")
print(f"   OK  torch {torch.__version__} (cuda), tensorrt {tensorrt.__version__}")
PY

# ======================================================================================
step "2/4 · python environment" "(~20 s first run, instant after)"
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
# Versions/sources are the proven ones from the working install:
#   - transformers 4.51.3 (a plain install pulls 5.x which drops torch-2.3 support)
#   - torch2trt pinned to the exact commit that built the validated engine
if "$PY" -c "import torch2trt, transformers, onnx" 2>/dev/null; then
  step "3/4 · conversion deps" "(already cached — skipping)"
  ok "transformers + torch2trt + onnx already present"
else
  step "3/4 · install conversion deps" "(~3–6 min FIRST RUN — torch2trt compiles from git)"
  echo "   installing transformers / onnx (pip wheels) ..."
  $PIP -q "numpy<2" "transformers==4.51.3" sentencepiece onnx onnxruntime onnx_graphsurgeon 2>&1 | indent
  echo "   building torch2trt from git (this is the slow part — please wait, not stalled) ..."
  TORCH2TRT_COMMIT="4e820ae31b4e35d59685935223b05b2e11d47b03"
  $PIP -q "git+https://github.com/NVIDIA-AI-IOT/torch2trt.git@${TORCH2TRT_COMMIT}" 2>&1 | indent
fi

# SAFETY: the installs above must NOT have disturbed the inherited CUDA torch / tensorrt.
"$PY" - <<'PY'
import sys, torch, tensorrt
if not torch.__version__.startswith("2.3"):
    sys.exit(f"FATAL: torch version changed to {torch.__version__} — install touched the CUDA torch!")
if not torch.cuda.is_available():
    sys.exit("FATAL: torch lost CUDA after install — aborting.")
print(f"   OK  torch {torch.__version__} (cuda), tensorrt {tensorrt.__version__} intact")
PY

# ======================================================================================
step "4/4 · build TensorRT engine" "(~2 min build + ~1.2GB model download on first run)"
echo "   converting $HF_MODEL"
echo "   (loads FP32 model -> torch2trt FP16 -> validates cos>=0.99 -> serializes ~940MB)"
echo "   (first run downloads the HF model ~1.2GB here — silent bar, please wait)"
set +e
HF_HUB_DISABLE_PROGRESS_BARS=1 "$PY" "$HERE/convert_siglip2.py" --hf-model "$HF_MODEL" --out "$OUT" 2>&1 | indent
RC=${PIPESTATUS[0]}      # python's real exit code, not sed's — the OOM(137) check depends on this
set -e
if [ "$RC" -ne 0 ]; then
  echo
  if [ "$RC" -eq 137 ] || [ "$RC" -eq 139 ]; then
    echo "FATAL: the engine build was Killed (exit $RC) — almost certainly an OUT-OF-MEMORY kill."
    echo "       It peaks ~12.5GB on the 16GB unified pool; something else is using too much RAM."
    mem_mitigations
  else
    echo "FATAL: engine build failed (exit $RC). See the error above."
  fi
  exit "$RC"
fi

# ======================================================================================
echo
echo "======================================================================"
echo "  DONE — engine ready"
echo "    $OUT"
echo "  next:  ./setup/setup_db.sh        # install faiss (quick)"
echo "         ./run_demo.sh              # build index + serve :8000"
echo "======================================================================"
