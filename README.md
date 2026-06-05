# CLIP-Based Surveillance Smart Search System

Semantic video search system for multi-camera surveillance. Search surveillance footage using natural language queries like "woman in red dress" instead of manually reviewing hours of video.

## Project Status (2026-06-01)

**Phase**: 2/3 — model selected & PN demo running; designing the production vector DB.  
**Deployed model**: **SigLIP2-L/16-256** (HuggingFace, TensorRT FP16 via torch2trt) — chosen after a full Jetson sweep. 384px is the future swap; the system is built model-swappable.  
**PN (Jetson Orin Nano)**: **3,145 img/min**, cos 0.999974 vs FP32, 2.3 GB — clears the 960 img/min target with headroom.  
**Done**: model decision, TRT conversion, repeatable install scripts (`scripts/setup_*.sh`), real-decode benchmark (~98 ms/img serial), end-to-end PN search demo (`web_pn/pn_app.py`).  
**Next**: production vector DB — time-sharded SQLite + FAISS IVFPQ, streaming ingest.  

**Jetson benchmark report**: See **[docs/JETSON_BENCHMARK_2026.md](./docs/JETSON_BENCHMARK_2026.md)** (the model-selection source of truth).

> This repo holds two tracks: a **desktop CLIP model-comparison tool** (`web/`, RTX GPU) used to choose the model, and the **PN deployment** (`web_pn/`, `scripts/`) targeting the Jetson. The RTX benchmarks below are from the desktop evaluation harness.

## Quick Start

### 1. Setup Environment

```bash
# Clone/navigate to project
cd clip_search

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install open-clip-torch pillow numpy tqdm
```

### 2. Test the Model

```bash
# Run basic model test
python scripts/test_model.py
```

**Expected output**: Model loads, benchmarks run, should meet 960 img/min requirement

### 3. Enable GPU (if not working)

```bash
# Load NVIDIA driver
sudo modprobe nvidia

# Verify
nvidia-smi

# Re-run test for GPU benchmarks
python scripts/test_model.py
```

## System Architecture

```
┌─────────────────────────────────────┐
│  GN (Gateway/NVR)                   │
│  - Records 32 cameras               │
│  - Generates thumbnails (2-sec)     │
│  - Web UI for search                │
└────────────┬────────────────────────┘
             │ Thumbnails (HTTP)
             ↓
┌─────────────────────────────────────┐
│  PN (Processing Node - Jetson)      │
│  - EVA-02-B/14 CLIP encoding        │
│  - FAISS vector database            │
│  - Search API                       │
└─────────────────────────────────────┘
```

## Performance Targets

| Metric | Target | Achieved (GPU) | Status |
|--------|--------|----------------|--------|
| **Throughput** | 960 img/min | **21,597 img/min** | ✅ **22× target** |
| **Single image** | <100ms | **5.6ms** | ✅ **18× faster** |
| **Batch of 16** | <1000ms | **44.5ms (2.8ms/img)** | ✅ **22× faster** |
| **Cameras supported** | 32 cameras @ 2-sec | **720 cameras** @ 2-sec | ✅ **22× capacity** |
| **Text query** | <50ms | **0.7ms** | ✅ **71× faster** |
| **Memory** | <8GB | **0.8GB** | ✅ **10× under budget** |

## Current Test Results

### ✅ GPU Performance (NVIDIA RTX 3090)

**EVA-02-B/16 on Linux PC with RTX 3090**:
- **Single image**: 5.6ms (179 img/sec)
- **Batch of 16**: 2.8ms/image = **21,597 img/min** ✅
- **Text query**: 0.7ms
- **Safety margin**: 2,150% above target (22× faster than needed!)

### CPU Baseline (for reference)

**EVA-02-B/16 on CPU**:
- Single image: 61.9ms
- Batch of 16: 56.7ms/image = 1,058 img/min
- **GPU Speedup**: 20× faster than CPU

**Full verification report**: [docs/MODEL_VERIFICATION_REPORT.md](./docs/MODEL_VERIFICATION_REPORT.md)

## Project Structure

```
clip_search/
├── README.md                    # This file
├── PROJECT_PLAN.md              # Detailed project plan
├── CLAUDE.md                    # Development guidelines
├── requirements.txt             # Python dependencies
├── docs/
│   ├── MODEL_COMPARISON_STUDY.md    # SigLIP vs EVA-02 analysis
│   └── MODEL_VERIFICATION_REPORT.md # GPU verification & benchmarks ✅
├── src/
│   ├── models/                  # Model loading & optimization
│   ├── api/                     # FastAPI endpoints
│   ├── encoding/                # Image/text encoding pipeline
│   ├── database/                # FAISS + SQLite
│   ├── preprocessing/           # Image preprocessing
│   └── search/                  # Vector search logic
├── scripts/
│   ├── test_model.py           # Model testing script ✓
│   └── README.md               # Scripts documentation
├── tests/                       # Unit & integration tests
└── configs/                     # Configuration files
```

## Documentation

- **[PROJECT_PLAN.md](./PROJECT_PLAN.md)**: Complete technical specification and implementation phases
- **[MODEL_COMPARISON_STUDY.md](./docs/MODEL_COMPARISON_STUDY.md)**: Detailed comparison of SigLIP vs EVA-02 models
- **[MODEL_VERIFICATION_REPORT.md](./docs/MODEL_VERIFICATION_REPORT.md)**: ✅ GPU verification and performance benchmarks
- **[ENCODING_AND_SEARCH_GUIDE.md](./docs/ENCODING_AND_SEARCH_GUIDE.md)**: Complete usage guide for encoding and searching
- **[CLIP_STRENGTHS_AND_IMPROVEMENTS.md](./docs/CLIP_STRENGTHS_AND_IMPROVEMENTS.md)**: CLIP capabilities, limitations, and accuracy improvement strategies
- **[VISION_MODEL_LANDSCAPE_2026.md](./docs/VISION_MODEL_LANDSCAPE_2026.md)**: Latest vision models, ChatGPT/VLM approach, and upgrade options
- **[CLIP_MODEL_COMPARISON_2026.md](./docs/CLIP_MODEL_COMPARISON_2026.md)**: Comprehensive CLIP model comparison with performance benchmarks and specifications 🆕
- **[CLAUDE.md](./CLAUDE.md)**: Project context and development guidelines
- **[scripts/README.md](./scripts/README.md)**: Testing scripts documentation

## Technology Stack

- **Deployed model**: SigLIP2-L/16-256 (HF `google/siglip2-large-patch16-256`, 1024-dim), TensorRT FP16 via torch2trt
- **Evaluation harness**: OpenCLIP (EVA-02-B/L, ViT-H/14, ViT-bigG/14, SigLIP2-B/L/SO400M) on the desktop tool
- **ML (PN)**: PyTorch 2.3 (Jetson CUDA build), `transformers==4.51.3`, `numpy<2`
- **Vector DB**: FAISS **CPU** (IndexFlatIP now → IndexIVFPQ + daily shards at scale)
- **API**: FastAPI + uvicorn
- **Database**: SQLite (metadata)
- **Platform**: Jetson Orin Nano 16GB (PN target), Linux PC + RTX (development/eval)

## Next Steps

### Done
- [x] Desktop model-comparison tool; 8 models indexed on CrowdHuman
- [x] Full Jetson benchmark sweep → **SigLIP2-L-256 selected**
- [x] TensorRT FP16 conversion (torch2trt) on the PN
- [x] Repeatable install scripts (`setup_model.sh`, `setup_db.sh`), clean-room tested
- [x] Real-decode benchmark on the PN (~98 ms/img serial)
- [x] End-to-end PN search demo (`web_pn/pn_app.py`)

### Active — production vector DB (designing now)
- [ ] SQLite metadata joined to FAISS via int64 IDs (`IndexIDMap2`)
- [ ] Daily time-shards; retention by dropping oldest shard files
- [ ] `IndexIVFPQ` compression at scale
- [ ] Streaming ingest concurrent with search (immutable history + active shard)

### Next
- [ ] Parallel ingest pipeline (HW decode / CPU resize / GPU encode)
- [ ] Search API with filtering (camera, time range, threshold)
- [ ] GN↔PN integration (thumbnail upload) + GN-side search UI

## Requirements

### Hardware
- **Development**: Linux PC with NVIDIA GPU (CUDA 12.x)
- **Production**: NVIDIA Jetson Orin Nano 16GB

### Software
- Python 3.12+
- CUDA 12.0+
- NVIDIA driver 550+
- 8GB+ disk space (for model weights)

### Python Dependencies
See `requirements.txt` for full list. Key packages:
- `torch>=2.0.0`
- `open-clip-torch>=2.20.0`
- `faiss-gpu>=1.7.4`
- `fastapi>=0.104.0`

## Usage Examples

### Load Model and Encode Image

```python
import open_clip
from PIL import Image

# Load model
model, _, preprocess = open_clip.create_model_and_transforms(
    'EVA02-B-16',
    pretrained='merged2b_s8b_b131k'
)

# Encode image
image = Image.open('thumbnail.jpg')
image_input = preprocess(image).unsqueeze(0)
image_features = model.encode_image(image_input)

# Encode text query
tokenizer = open_clip.get_tokenizer('EVA02-B-16')
text = tokenizer(["woman in red dress"])
text_features = model.encode_text(text)

# Compute similarity
similarity = (image_features @ text_features.T)
print(f"Similarity: {similarity.item():.3f}")
```

### Run Benchmarks

```bash
# Full model test
python scripts/test_model.py

# Expected output:
# ✓ Model loads successfully
# ✓ Benchmarks complete
# ✓ Meets 960 img/min requirement
```

## Troubleshooting

### GPU Not Detected
```bash
# Check GPU
lspci | grep -i nvidia

# Load driver
sudo modprobe nvidia

# Verify
nvidia-smi
```

### CUDA Out of Memory
- Reduce batch size in test script
- Close other GPU applications
- Try CPU mode (slower but works)

### Model Download Fails
- Check internet connection
- Set HuggingFace token: `export HF_TOKEN=your_token`
- Try different mirror: `HF_ENDPOINT=https://hf-mirror.com`

## Contributing

This is a private project for GN surveillance system integration.

## License

Proprietary - Internal use only

## Contact

Project Lead: chinghokuk@gmail.com  
Last Updated: 2026-06-01
