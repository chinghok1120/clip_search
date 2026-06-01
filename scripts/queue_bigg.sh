#!/usr/bin/env bash
# Wait for the in-flight DFN-H encode to finish, then encode bigG (fp16).
set -u
cd "$(dirname "$0")/.."
source venv/bin/activate

echo "[queue] waiting for DFN-H encode to finish..."
while pgrep -f "encode_images.py.*ViT-H-14-378-quickgelu" >/dev/null; do
    sleep 30
done

if [ ! -f embeddings/crowdhuman_vit-h-14-378-quickgelu.faiss ]; then
    echo "[queue] ABORT: DFN-H output not found; not starting bigG." >&2
    exit 1
fi
echo "[queue] DFN-H done. Starting bigG (fp16)..."

python scripts/encode_images.py \
    --input ~/datasets/crowdhuman/train/images \
    --output embeddings/crowdhuman \
    --model ViT-bigG-14 \
    --pretrained laion2b_s39b_b160k \
    --device cuda \
    --batch-size 32 \
    --precision fp16

echo "[queue] bigG encode finished."
