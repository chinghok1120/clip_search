#!/bin/bash
# Quick test script

echo "=== Testing Image Encoding & Search ==="
echo ""

# Check if we have test images
if [ ! -d ~/datasets ]; then
  echo "No datasets folder found. Create a test folder with some images:"
  echo "  mkdir -p ~/test_images"
  echo "  # Copy some .jpg or .png files to ~/test_images"
  exit 1
fi

# Find a folder with images
IMAGE_DIR=$(find ~/datasets -name "*.jpg" -o -name "*.png" | head -1 | xargs dirname)

if [ -z "$IMAGE_DIR" ]; then
  echo "No images found in ~/datasets"
  exit 1
fi

echo "Found images in: $IMAGE_DIR"
echo ""

# Count images
NUM_IMAGES=$(find "$IMAGE_DIR" -maxdepth 1 \( -name "*.jpg" -o -name "*.png" \) | wc -l)
echo "Number of images: $NUM_IMAGES"
echo ""

# Activate venv
source venv/bin/activate

# Encode
echo "=== Step 1: Encoding images ==="
python scripts/encode_images.py \
  --input "$IMAGE_DIR" \
  --output ./test_embeddings \
  --batch-size 16

echo ""
echo "=== Step 2: Search test ==="
echo "Query: 'person'"
python scripts/search_images.py \
  --embeddings ./test_embeddings \
  --query "person" \
  --top-k 3

echo ""
echo "✓ Test complete!"
