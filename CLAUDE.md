# CLIP-Based Surveillance Search System

This project implements a semantic video search system for multi-camera surveillance using CLIP models. Users can search surveillance footage using natural language queries (e.g., "woman in red dress") instead of manually scrubbing through hours of video.

## Project Context

**Main Documentation**: See [PROJECT_PLAN.md](./PROJECT_PLAN.md) for complete technical specifications, implementation phases, and architecture details.

### System Overview
- **GN (Gateway/NVR)**: Network Video Recorder that handles 8-32 cameras, generates thumbnails every 2 seconds, provides web interface
- **PN (Processing Node)**: Jetson Orin Nano 16GB that encodes thumbnails using CLIP, maintains vector database, performs semantic search
- **User Flow**: Natural language query → CLIP text encoding → vector similarity search → return matching thumbnails with timestamps

### Key Architectural Decisions
1. **Text encoding happens on PN** (GN has no GPU)
2. **Vector database stored on PN** (initially - may move to GN later)
3. **Model: SigLIP2-L/16-256** (HuggingFace weights, TensorRT FP16 via torch2trt) — selected over EVA-02 after a full Jetson benchmark sweep (see [docs/JETSON_BENCHMARK_2026.md](./docs/JETSON_BENCHMARK_2026.md)). **SigLIP2-L/16-384** is the planned future swap. The system is built model-swappable (single `PROFILE` dict; main swap cost is re-indexing).
4. **Communication: REST API** between GN and PN
5. **Vector DB: FAISS (CPU-first)** — `IndexFlatIP` at demo scale, `IndexIVFPQ` + daily time-shards at production scale. GPU search is a later option (GPU is reserved for encoding).

## Technical Stack

### PN (Processing Node)
- **Hardware**: NVIDIA Jetson Orin Nano 16GB (host `PNServer`, `superrx@210.17.139.83`)
- **Model**: SigLIP2-L/16-256 (HuggingFace `google/siglip2-large-patch16-256`), 1024-dim embeddings
- **Optimization**: TensorRT FP16 via **torch2trt** (eager attention) — 3,145 img/min, cos 0.999974 vs FP32. INT8 ruled out on this board (§7/§12 of benchmark doc).
- **Vector Search**: FAISS **CPU** (IndexFlatIP now → IndexIVFPQ + daily shards for scale)
- **Metadata DB**: SQLite (camera_id, timestamp, thumbnail_path) joined to FAISS via int64 IDs
- **API**: FastAPI + uvicorn
- **Language**: Python 3.10 (PN venv with `--system-site-packages` for CUDA torch); 3.8+ elsewhere
- **Pinned deps**: `transformers==4.51.3`, `numpy<2` (torch 2.3 compatibility)

### Key Libraries
```
transformers==4.51.3  # SigLIP2 model + tokenizer (HF); pinned for torch 2.3
torch                 # PyTorch inference
torch2trt             # TensorRT conversion (NVIDIA-AI-IOT) — the working SigLIP TRT path
tensorrt              # Model optimization for Jetson
faiss-cpu             # Vector similarity search (CPU-first)
fastapi               # REST API framework
pillow                # Image preprocessing
numpy<2               # Array operations (torch 2.3 compatibility)
sqlite3               # Metadata storage
open_clip_torch       # used on the desktop model-comparison tool (EVA-02, ViT-H/bigG, SigLIP2)
```

### GN (Gateway/NVR)
- Thumbnail generation from video streams (640×360 JPEG, 2-second interval)
- HTTP client to send thumbnails to PN
- Web interface for search (text input, thumbnail grid display)
- Video playback integration (click thumbnail → jump to timestamp)

## Performance Requirements

### Throughput
- **960 images/minute** sustained encoding (32 cameras × 30 thumbnails/min)
- Target: <100ms per image (single), ~50ms per image (batch of 16)

### Search Latency
- Text encoding: <50ms
- Vector search: <500ms (up to 1M embeddings)
- End-to-end: <1 second

### Resource Limits
- GPU memory: <8GB (TensorRT optimized model + FAISS)
- Storage: ~9.3GB/week for embeddings, ~186GB/week for thumbnails (7-day retention)

## Development Workflow

### Model Development
1. Load EVA-02-L/14 from OpenCLIP
2. Export to ONNX format
3. Convert to TensorRT (FP16 first, INT8 if needed)
4. Benchmark: throughput, latency, memory usage
5. Iterate on optimization (batch size, precision, layer fusion)

### API Development
- Use FastAPI for REST endpoints
- Implement batch encoding for efficiency
- Add health checks and metrics endpoints
- Test with mock thumbnail data before GN integration

### Database Development
- Start with FAISS IndexFlatIP (exact search, simple)
- Migrate to IndexIVFPQ when embeddings exceed 100K
- SQLite for metadata (sufficient for single PN)
- Consider PostgreSQL if multiple PNs in future

### Testing
- Unit tests for preprocessing, encoding, search logic
- Integration tests for end-to-end flows
- Performance tests with realistic workloads (960 images/min)
- Accuracy tests with curated surveillance image queries

## Code Organization

```
clip_search/
├── PROJECT_PLAN.md          # Detailed project specification
├── CLAUDE.md                # This file
├── src/
│   ├── models/              # Model loading, TensorRT conversion
│   ├── api/                 # FastAPI endpoints
│   ├── encoding/            # Image/text encoding pipeline
│   ├── database/            # FAISS + SQLite interfaces
│   ├── preprocessing/       # Image preprocessing (resize, normalize)
│   └── search/              # Vector search and ranking logic
├── scripts/
│   ├── benchmark.py         # Performance benchmarking
│   ├── export_onnx.py       # Model export utilities
│   └── index_builder.py     # FAISS index creation/migration
├── tests/
│   ├── test_encoding.py
│   ├── test_search.py
│   └── test_integration.py
├── configs/
│   ├── model_config.yaml    # Model parameters
│   └── api_config.yaml      # API settings
└── docs/
    ├── api.md               # API documentation
    └── deployment.md        # Deployment guide
```

## Development Phases

See [PROJECT_PLAN.md](./PROJECT_PLAN.md) for detailed phase breakdown. Summary:

1. **Phase 1**: Model setup and TensorRT optimization on Jetson — ✅ **DONE** (SigLIP2-L-256 selected; full sweep in [docs/JETSON_BENCHMARK_2026.md](./docs/JETSON_BENCHMARK_2026.md))
2. **Phase 2**: Thumbnail encoding pipeline and storage — 🔄 **IN PROGRESS** (PN demo runs end-to-end on a static CrowdHuman index; production DB layer is the active design topic — see below)
3. **Phase 3**: Search backend and vector search optimization — 🔄 query path works on the PN (`pn/web_pn/pn_app.py`); IVFPQ/sharding pending
4. **Phase 4**: GN-PN integration and thumbnail streaming — ⬜ not started
5. **Phase 5**: Web UI for search and video playback — ⬜ desktop comparison UI exists (`web/`); GN-side UI not started
6. **Phase 6**: Production optimization and monitoring — ⬜ not started

Current Phase: **Phase 2/3 — designing the production vector DB (time-sharded SQLite + FAISS IVFPQ, streaming ingest).** Model selection, PN feasibility, TRT conversion, install scripts, and an end-to-end PN search demo are all complete.

> **Note on repo scope:** this repo currently serves two things — (1) a **desktop CLIP model-comparison tool** (`web/`, RTX GPU) used to pick the model, and (2) the **PN deployment** (`pn/` — maps 1:1 to `PN:~/clip_search/`) targeting the Jetson. CLAUDE.md/PROJECT_PLAN describe the surveillance product; the desktop tool is the evaluation harness that fed the model decision.

## Key Technical Constraints

### Hardware Constraints
- **Jetson Orin Nano**: 16GB unified memory (shared CPU/GPU)
- **GPU**: 1024 CUDA cores, Ampere architecture
- **Power**: 7-15W power budget
- **Storage**: NVMe SSD required for fast I/O

### Model Constraints
- Must fit in <8GB GPU memory after TensorRT optimization
- Must sustain 960 images/min encoding throughput
- Text encoding must be <50ms per query

### Scale Constraints
- Single PN handles up to 32 cameras (with 2-second sampling)
- Vector search must scale to millions of embeddings
- Network bandwidth GN→PN must support thumbnail streaming

## Important Considerations

### Security
- Thumbnails may contain sensitive surveillance data
- Implement authentication for PN API
- Consider encryption for GN↔PN communication
- Access control for search queries (audit logs)

### Privacy
- Embeddings are privacy-sensitive (can reconstruct scenes)
- Implement retention policies for thumbnails and embeddings
- Consider anonymization for demo/testing

### Reliability
- Handle GN↔PN network failures gracefully
- Implement retry logic with exponential backoff
- Monitor PN health (GPU temp, memory usage, disk space)
- Thumbnail queue to buffer during PN downtime

### Accuracy
- CLIP is zero-shot but may miss domain-specific concepts
- Consider fine-tuning on surveillance imagery
- Provide query suggestions and examples to users
- Log failed searches to identify accuracy gaps

## Future Enhancements

- Multi-modal search (text + reference image)
- Temporal queries ("person entering 2-3 PM")
- Cross-camera object tracking
- Real-time alerts on matching events
- Fine-tuning on surveillance-specific data
- Distributed PNs for >32 cameras
- Mobile app for remote access

## Common Commands

```bash
# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run API server (PN)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Benchmark model
python scripts/benchmark.py --model eva02_large --batch-size 16

# Build FAISS index from embeddings
python scripts/index_builder.py --input embeddings/ --output index.faiss

# Run tests
pytest tests/ -v

# Export model to TensorRT
python scripts/export_onnx.py --model eva02_large --output models/eva02.onnx
trtexec --onnx=models/eva02.onnx --saveEngine=models/eva02.trt --fp16
```

## Debugging Tips

- **Low throughput**: Check batch size, TensorRT optimization, GPU utilization
- **High memory usage**: Reduce batch size, enable INT8 quantization, profile memory
- **Slow search**: Check FAISS index type (migrate to IVF), profile query
- **Poor accuracy**: Review preprocessing (normalization), check model weights, add query examples
- **API errors**: Check logs in `/var/log/clip_search/`, verify model loaded correctly

## References

- Project Plan: [PROJECT_PLAN.md](./PROJECT_PLAN.md)
- EVA-02 Paper: https://arxiv.org/abs/2303.11331
- OpenCLIP: https://github.com/mlfoundations/open_clip
- FAISS Docs: https://github.com/facebookresearch/faiss/wiki
- Jetson Orin: https://developer.nvidia.com/embedded/jetson-orin
- TensorRT: https://developer.nvidia.com/tensorrt

## Contact

Project Owner: chinghokuk@gmail.com  
Last Updated: 2026-06-05
