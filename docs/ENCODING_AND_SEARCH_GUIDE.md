# Image Encoding and Search Guide

Quick guide for encoding images and searching with natural language queries.

---

## Overview

The workflow has two steps:

1. **Encode**: Process images folder → create embeddings
2. **Search**: Query with text → get matching images

```
Images Folder → [encode_images.py] → Embeddings + Index → [search_images.py] → Results
```

---

## Step 1: Encode Images

### Basic Usage

```bash
cd ~/projects/clip_search
source venv/bin/activate

python scripts/encode_images.py \
  --input /path/to/images \
  --output /path/to/embeddings
```

### Full Options

```bash
python scripts/encode_images.py \
  --input /path/to/images \         # Input folder with images
  --output /path/to/embeddings \    # Output folder
  --batch-size 16 \                 # Batch size (default: 16)
  --device cuda \                   # Device: cuda or cpu
  --model EVA02-B-16 \              # Model name
  --format faiss                    # Output format: faiss or numpy
```

### Output Files

```
embeddings/
├── embeddings.index     # FAISS index (fast search)
└── metadata.json        # Image paths and info
```

### Example

```bash
# Encode surveillance thumbnails
python scripts/encode_images.py \
  --input ~/datasets/surveillance_thumbnails \
  --output ~/datasets/surveillance_embeddings \
  --batch-size 16

# Output:
# Loading model: EVA02-B-16
# Device: cuda
# ✓ Model loaded, embedding dimension: 512
# Scanning for images in: ~/datasets/surveillance_thumbnails
# ✓ Found 1,000 images
# Encoding: 100%|████████| 63/63 [00:05<00:00, 12.5it/s]
# ================================================================================
# ENCODING STATISTICS
# ================================================================================
#
# 📊 Input:
#   Images found:        1,000
#   Images processed:    1,000
#   Failed:              0
#   Success rate:        100.0%
#
# ⚙️  Configuration:
#   Model:               EVA02-B-16
#   Device:              cuda
#   Batch size:          16
#   Embedding dimension: 512
#
# ⏱️  Performance:
#   Encoding time:       2.77s
#   Saving time:         0.12s
#   Total time:          2.89s
#   Time per image:      2.77ms
#   Throughput:          361.0 img/sec
#   Throughput:          21,659 img/min
#
# 💾 Output:
#   Index file:          ~/datasets/surveillance_embeddings/embeddings.index
#   Index size:          1.95 MB
#   Metadata file:       ~/datasets/surveillance_embeddings/metadata.json
#   Metadata size:       156.32 KB
#   Total size:          2.10 MB
#   Bytes per embedding: 2102 bytes
#
# ✅ ENCODING COMPLETE
# ================================================================================
```

---

## Step 2: Search Images

### Basic Usage

```bash
python scripts/search_images.py \
  --embeddings /path/to/embeddings \
  --query "woman in red dress"
```

### Full Options

```bash
python scripts/search_images.py \
  --embeddings /path/to/embeddings \  # Embeddings folder
  --query "woman in red dress" \      # Text query
  --top-k 10 \                        # Number of results
  --device cuda \                     # Device: cuda or cpu
  --model EVA02-B-16 \                # Model (must match encoding)
  --show-full-path \                  # Show full paths
  --json                              # Output as JSON
```

### Example

```bash
# Search for "person in red jacket"
python scripts/search_images.py \
  --embeddings ~/datasets/surveillance_embeddings \
  --query "person in red jacket" \
  --top-k 5

# Output:
# ================================================================================
# INITIALIZING SEARCH
# ================================================================================
# Loading model: EVA02-B-16
# ✓ Model loaded
# Loading FAISS index: ~/datasets/surveillance_embeddings/embeddings.index
# ✓ Loaded 1000 embeddings
# Loading metadata: ~/datasets/surveillance_embeddings/metadata.json
# ✓ Loaded metadata for 1000 images
#
# ================================================================================
# SEARCHING
# ================================================================================
# Query: 'person in red jacket'
# Top-K: 5
#
# ================================================================================
# SEARCH STATISTICS
# ================================================================================
#
# 📝 Query:
#   Text:                'person in red jacket'
#   Top-K requested:     5
#   Results returned:    5
#
# 💾 Database:
#   Type:                FAISS
#   Total embeddings:    1,000
#   Embedding dimension: 512
#   Index file:          ~/datasets/surveillance_embeddings/embeddings.index
#   Index size:          1.95 MB
#   Metadata file:       ~/datasets/surveillance_embeddings/metadata.json
#   Metadata size:       156.32 KB
#   Total size:          2.10 MB
#   Bytes per embedding: 2102 bytes
#
# ⏱️  Performance:
#   Text encoding:       0.72ms
#   Vector search:       1.15ms
#   Total time:          1.87ms
#
# 📊 Results:
#   Top score:           87.3%
#   Lowest score:        79.1%
#   Average score:       83.6%
#
# ⚙️  Configuration:
#   Model:               EVA02-B-16
#   Device:              cuda
#
# ================================================================================
#
# ================================================================================
#   Search Results (5 matches)
# ================================================================================
# 
# #1  Score: 87.3%
#   File: cam2_20260512_143045.jpg
#   Size: (640, 360)
# 
# #2  Score: 84.1%
#   File: cam5_20260512_150230.jpg
#   Size: (640, 360)
# 
# #3  Score: 82.5%
#   File: cam1_20260512_141522.jpg
#   Size: (640, 360)
# 
# #4  Score: 80.9%
#   File: cam3_20260512_145617.jpg
#   Size: (640, 360)
# 
# #5  Score: 79.1%
#   File: cam2_20260512_150045.jpg
#   Size: (640, 360)
#
# ================================================================================
```

---

## Storage Comparison

### Option 1: FAISS + JSON (Default) ⭐

**Pros:**
- ✅ Fast search (O(log n) with IVF indexes)
- ✅ GPU-accelerated search available
- ✅ Scalable to millions of images
- ✅ Incremental updates possible

**Cons:**
- ⚠️ Requires FAISS library

**File Structure:**
```
embeddings/
├── embeddings.index    # Binary FAISS index
└── metadata.json       # JSON with image paths
```

**When to use:** Production, large datasets (>10K images)

### Option 2: NumPy (Simple)

**Pros:**
- ✅ Simple, no extra dependencies
- ✅ Easy to inspect/debug

**Cons:**
- ❌ Slow search (O(n) linear scan)
- ❌ No GPU acceleration for search
- ❌ Loads entire array into memory

**File Structure:**
```
embeddings/
└── embeddings.npz      # Contains embeddings + metadata
```

**When to use:** Small datasets (<10K images), testing

---

## Performance

### Encoding Performance (RTX 3090)

| Batch Size | Speed (img/sec) | Throughput (img/min) |
|------------|----------------|---------------------|
| 1 | 122 | 7,308 |
| 4 | 322 | 19,309 |
| 8 | 344 | 20,655 |
| **16** | **360** | **21,597** |

**Recommendation:** Use batch-size 16 for maximum throughput

### Search Performance

| Operation | FAISS (GPU) | FAISS (CPU) | NumPy (CPU) |
|-----------|------------|-------------|-------------|
| **Text encoding** | ~1ms | ~10ms | ~10ms |
| **Vector search (10K)** | ~1ms | ~5ms | ~50ms |
| **Vector search (100K)** | ~5ms | ~20ms | ~500ms |
| **Vector search (1M)** | ~20ms | ~100ms | ~5000ms |

**Recommendation:** Use FAISS for >10K images

---

## Advanced Usage

### 1. Encode Multiple Folders

```bash
# Encode camera 1 thumbnails
python scripts/encode_images.py \
  --input /recordings/cam1/thumbnails \
  --output /embeddings/cam1

# Encode camera 2 thumbnails
python scripts/encode_images.py \
  --input /recordings/cam2/thumbnails \
  --output /embeddings/cam2

# Later: merge indexes (requires custom script)
```

### 2. JSON Output for Integration

```bash
# Get results as JSON for API integration
python scripts/search_images.py \
  --embeddings /embeddings \
  --query "delivery truck" \
  --top-k 20 \
  --json > results.json
```

### 3. Batch Search Queries

```bash
# Create query list
cat > queries.txt <<EOF
person in red dress
blue car in parking lot
delivery truck at loading dock
person with umbrella
EOF

# Search each query
while read query; do
  echo "Query: $query"
  python scripts/search_images.py \
    --embeddings /embeddings \
    --query "$query" \
    --top-k 5
done < queries.txt
```

### 4. Filter by Score Threshold

```bash
# Get only high-confidence matches (>80%)
python scripts/search_images.py \
  --embeddings /embeddings \
  --query "woman in red dress" \
  --top-k 100 \
  --json | jq '.[] | select(.score > 0.8)'
```

---

## Integration with GN/PN

### On PN (Processing Node)

**1. Create embeddings from new thumbnails:**

```bash
# Cron job: encode new thumbnails every minute
*/1 * * * * python /opt/clip_search/scripts/encode_images.py \
  --input /recordings/new_thumbnails \
  --output /embeddings \
  --batch-size 16
```

**2. Expose search API:**

See `src/api/main.py` for FastAPI implementation (Phase 2)

### On GN (Gateway/NVR)

**Web interface sends search queries to PN via HTTP:**

```python
import requests

response = requests.post('http://pn:8000/search', json={
    'query': 'woman in red dress',
    'top_k': 20,
    'filters': {
        'camera_ids': [1, 2, 3],
        'start_time': '2026-05-12T08:00:00',
        'end_time': '2026-05-12T18:00:00'
    }
})

results = response.json()
# Display thumbnails in web UI
```

---

## Troubleshooting

### "FAISS not installed"

```bash
# Install FAISS CPU version
pip install faiss-cpu

# Or for GPU version (Jetson/CUDA)
pip install faiss-gpu
```

### "CUDA out of memory"

Reduce batch size:
```bash
python scripts/encode_images.py --batch-size 8  # or 4
```

### "No images found"

Check supported formats:
- Supported: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`
- Case insensitive
- Searches recursively in subfolders

### Slow search on large dataset

Use FAISS IVF index for >100K embeddings (requires custom code):
```python
import faiss
# Instead of IndexFlatIP:
index = faiss.IndexIVFFlat(quantizer, dim, nlist)
```

---

## Tips & Best Practices

### For Production

1. ✅ **Use FAISS** (not NumPy) for fast search
2. ✅ **Batch encode** with size 16 for best throughput
3. ✅ **Normalize embeddings** (already done in scripts)
4. ✅ **Use IndexIVFPQ** for >1M embeddings
5. ✅ **Monitor GPU memory** (~0.8GB for model)

### For Development

1. ✅ Start with small dataset (100-1000 images)
2. ✅ Test different queries to validate accuracy
3. ✅ Use `--json` output for easy parsing
4. ✅ Profile encoding speed with your actual images

### Query Writing Tips

**Good queries** (specific, visual):
- ✅ "person in red jacket"
- ✅ "blue sedan in parking lot"
- ✅ "delivery truck at loading dock"
- ✅ "person carrying umbrella"

**Bad queries** (abstract, non-visual):
- ❌ "suspicious person" (abstract concept)
- ❌ "person from yesterday" (temporal)
- ❌ "person ID 12345" (requires face recognition)
- ❌ "license plate ABC123" (requires OCR)
- ❌ "person smoking cigarette" (fine-grained action)

**For detailed explanation of what works and what doesn't, see:**
- [CLIP Strengths and Improvements Guide](./CLIP_STRENGTHS_AND_IMPROVEMENTS.md)

---

## Next Steps

1. **Phase 2:** Build REST API service (`src/api/`)
2. **Phase 3:** Add filtering (camera ID, timestamp, confidence)
3. **Phase 4:** Integrate with GN web interface
4. **Phase 5:** Deploy to Jetson Orin Nano
5. **Phase 6:** Optimize with TensorRT

---

## References

- **Scripts**: `scripts/encode_images.py`, `scripts/search_images.py`
- **Model**: EVA-02-B/14 from OpenCLIP
- **FAISS**: https://github.com/facebookresearch/faiss
- **Verification**: `docs/MODEL_VERIFICATION_REPORT.md`
