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
HF_MODEL="${HF_MODEL:-google/siglip2-large-patch16-256}"
OUT="${OUT:-$HOME/clip_search/siglip2_l_256_hf_fp16.pth}"
TARGET="${TARGET:-venv}"

# 0. resolve install target -> $PY (python) and $PIP (pip install command)
if [ "$TARGET" = "user" ]; then
  echo ">> TARGET=user: installing into ~/.local (user scope), no venv"
  PY="python3"
  PIP="python3 -m pip install --user"
else
  VENV="${VENV:-$HOME/clip_search/venv}"
  if [ ! -x "$VENV/bin/python" ]; then
    echo ">> creating venv $VENV (--system-site-packages to inherit JetPack torch/tensorrt)"
    python3 -m venv --system-site-packages "$VENV"
  fi
  PY="$VENV/bin/python"
  PIP="$VENV/bin/pip install"
  echo ">> TARGET=venv: using $VENV"
fi

# 1. PRE-BUILD GUARD (flexible): hard-fail only on genuine capability gaps, warn on version drift.
#    Philosophy: require CAPABILITIES, treat VERSIONS as advisory. Correctness is gated AFTER the
#    build by convert_siglip2.py's cos>=0.99 check — so this stays portable to newer PNs.
echo ">> preflight checks"
command -v git >/dev/null 2>&1 || { echo "FATAL: 'git' is required (torch2trt installs from git)."; exit 1; }
FREE_GB=$(df -Pk "$HOME" | awk 'NR==2{print int($4/1048576)}')
if [ "${FREE_GB:-0}" -lt 5 ]; then
  echo "FATAL: only ${FREE_GB}GB free in \$HOME; need >=5GB (deps + ~1.2GB model + 940MB engine)."; exit 1
fi
echo "   disk: ${FREE_GB}GB free (>=5GB) OK"
"$PY" - <<'PY'
import sys
# HARD: capabilities that genuinely block a GPU engine build
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
# SOFT: version drift is advisory only (correctness is gated post-build by cos>=0.99)
if not torch.__version__.startswith("2.3"):
    print(f"   WARN: torch {torch.__version__} != validated 2.3.x — proceeding; if the pinned "
          f"torch2trt commit fails to export, a torch-matched commit may be needed.")
if not str(tensorrt.__version__).startswith("10.3"):
    print(f"   WARN: tensorrt {tensorrt.__version__} != validated 10.3.x — proceeding.")
print(f"   preflight OK: torch {torch.__version__} (cuda), tensorrt {tensorrt.__version__}")
PY

# 2. install the conversion layer (numpy<2 held for torch 2.3).
#    Versions/sources are the proven ones from the working install:
#      - transformers 4.51.3 (a plain install pulls 5.x which drops torch-2.3 support)
#      - torch2trt pinned to the exact commit that built the validated engine
echo ">> installing conversion deps [TARGET=$TARGET]"
$PIP -q "numpy<2" "transformers==4.51.3" sentencepiece onnx onnxruntime onnx_graphsurgeon
TORCH2TRT_COMMIT="4e820ae31b4e35d59685935223b05b2e11d47b03"
$PIP -q "git+https://github.com/NVIDIA-AI-IOT/torch2trt.git@${TORCH2TRT_COMMIT}"

# SAFETY: the installs above must NOT have disturbed the inherited CUDA torch / tensorrt.
"$PY" - <<'PY'
import sys, torch, tensorrt
if not torch.__version__.startswith("2.3"):
    sys.exit(f"FATAL: torch version changed to {torch.__version__} — install touched the CUDA torch!")
if not torch.cuda.is_available():
    sys.exit("FATAL: torch lost CUDA after install — aborting.")
print(f"   safety OK: torch {torch.__version__} (cuda), tensorrt {tensorrt.__version__} intact")
PY

# 3. build + serialize + validate the engine (convert_siglip2.py gates cos>=0.99)
echo ">> converting $HF_MODEL -> $OUT"
"$PY" "$HERE/convert_siglip2.py" --hf-model "$HF_MODEL" --out "$OUT"

echo ">> model setup done: $OUT"
