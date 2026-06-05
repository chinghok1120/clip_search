# CLIP-Based Surveillance Smart Search System
## Project Plan

**Release:** v0.2.0  
**Plan revision:** 2.0  
**Last Updated:** 2026-06-05  
**Status:** Phase 2/3 — model selected & PN demo running; designing the production vector DB

---

## Current Status (2026-06-01)

**Model selection and PN feasibility are complete.** What's done:
- **Model decided: SigLIP2-L/16-256** (HuggingFace weights, TensorRT FP16 via torch2trt) — **3,145 img/min, cos 0.999974** vs FP32 on the Jetson Orin Nano. **SigLIP2-L/16-384** (1,151 img/min) is the planned future swap. Chosen after a full EVA-02 + SigLIP2 sweep — see **[docs/JETSON_BENCHMARK_2026.md](./docs/JETSON_BENCHMARK_2026.md)**.
- **Desktop model-comparison tool** (`web/`) used to compare 8 indexed models on CrowdHuman and drive the accuracy decision.
- **PN end-to-end demo running** (`pn/web_pn/pn_app.py`): TRT SigLIP2-256 image index + HF text-tower query path, LAN-accessible. Model-swappable via a single `PROFILE` dict.
- **Real-decode benchmark on the PN**: ~98 ms/image serial (decode+resize+encode); production will parallelize across HW-decode / CPU-resize / GPU-encode.
- **Repeatable production install scripts** (`pn/setup/setup_model.sh`, `pn/setup/setup_db.sh`), clean-room tested, with a `TARGET=user` (non-venv) option.
- **FAISS: CPU-first** (`IndexFlatIP` at demo scale).

**Active work (next):** the production vector-DB layer — SQLite metadata joined to FAISS via int64 IDs (`IndexIDMap2`), **daily time-shards**, `IndexIVFPQ` compression at scale, retention by dropping oldest shards, and streaming ingest concurrent with search (immutable history shards + one writable active shard). Design under discussion.

**Not started:** GN↔PN integration/API, GN-side search UI, monitoring.

---

## Executive Summary

Building a semantic video search system for multi-camera surveillance using CLIP. The system enables users to search hours of surveillance footage using natural language queries like "woman in red dress" by encoding camera thumbnails and matching them against text embeddings. **Deployed model: SigLIP2-L/16-256** (the original plan targeted EVA-02-L/14; the Jetson benchmark sweep moved the decision to SigLIP2 — see Current Status above).

### System Overview
- **GN (Gateway/NVR)**: Records 8-32 cameras (H.264/H.265), generates thumbnails every 2 seconds per camera (640×360 JPEG), provides web interface for search
- **PN (Processing Node)**: Jetson Orin Nano 16GB that encodes thumbnails using CLIP, stores embeddings, performs vector search
- **User Flow**: User enters text query → PN encodes text → vector search → returns matching thumbnails with timestamps

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                           GN (Gateway/NVR)                   │
│  - Records 8-32 cameras (H.264/H.265)                       │
│  - Generates thumbnails (640×360 JPEG, 2-second interval)   │
│  - Stores video + thumbnails on HDD                         │
│  - Web server for user interface                            │
│  - Sends thumbnails to PN via HTTP/gRPC                     │
│  - Displays search results (thumbnails + timestamps)        │
└───────────────────────┬─────────────────────────────────────┘
                        │ Thumbnails
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              PN (Processing Node - Jetson Orin Nano)        │
│  - Receives thumbnails from GN                              │
│  - CLIP image encoding (EVA-02-L/14 + TensorRT)            │
│  - CLIP text encoding (for user queries)                    │
│  - Vector database (FAISS-GPU + metadata DB)                │
│  - Vector search API                                         │
│  - Returns top-K matching results                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Technical Stack

### PN (Processing Node)
- **Hardware**: NVIDIA Jetson Orin Nano (16GB RAM, 1024-core Ampere GPU). **Must run MAXN + `jetson_clocks`** (default 25W mode halves the GPU clock → ~2× slower).
- **Model**: SigLIP2-L/16-256 (HuggingFace `google/siglip2-large-patch16-256`), 1024-dim embeddings. 384px variant is the future swap.
- **Inference**: TensorRT **FP16 via torch2trt** (eager attention). INT8 ruled out on this board (accuracy collapse + no speedup — see benchmark §7/§12).
- **Vector Database**: FAISS **CPU** (`IndexFlatIP` → `IndexIVFPQ` + daily shards for scale)
- **Metadata Database**: SQLite (camera_id, timestamp, thumbnail_path), joined to FAISS by int64 ID
- **API Framework**: FastAPI + uvicorn
- **Language**: Python 3.10 (PN venv built `--system-site-packages` to inherit CUDA torch 2.3); pinned `transformers==4.51.3`, `numpy<2`

### GN (Gateway/NVR)
- **Thumbnail Generation**: Extract from video stream every 2 seconds
- **Communication**: HTTP REST API or gRPC client
- **Web Interface**: Search UI with thumbnail display
- **Storage**: HDD for video files and thumbnails

### Key Libraries
- `open_clip_torch` - EVA-02-L/14 model
- `torch` + `tensorrt` - Model optimization
- `faiss-gpu` - Vector similarity search
- `fastapi` + `uvicorn` - REST API
- `pillow` / `opencv-python` - Image processing
- `numpy` - Array operations
- `sqlite3` / `sqlalchemy` - Metadata storage

---

## Implementation Phases

### Phase 1: Model Setup & Optimization (Week 1-2) — ✅ COMPLETE
**Goal**: Get the encoder running efficiently on Jetson Orin Nano
**Outcome**: SigLIP2-L-256 on TRT FP16 (torch2trt), 3,145 img/min, cos 0.999974. Full sweep + TRT/INT8 findings in [docs/JETSON_BENCHMARK_2026.md](./docs/JETSON_BENCHMARK_2026.md).

- [ ] Set up Jetson Orin Nano development environment
- [ ] Install PyTorch, TensorRT, CUDA dependencies
- [ ] Load EVA-02-L/14 OpenCLIP model
- [ ] Export model to ONNX format
- [ ] Convert ONNX to TensorRT (FP16 optimization)
- [ ] Benchmark inference time (target: <100ms per image)
- [ ] Implement batch processing (8-16 images)
- [ ] Test text encoding performance

**Deliverables**:
- TensorRT-optimized image encoder
- TensorRT-optimized text encoder
- Benchmark report (throughput, latency, memory usage)

### Phase 2: Embedding Pipeline (Week 3-4) — 🔄 IN PROGRESS
**Goal**: Build thumbnail ingestion and encoding service
**Status**: offline batch encode + index build work on the PN (static CrowdHuman demo); the **streaming ingest + production storage schema below is the active design**. Real-decode benchmarked at ~98 ms/image serial; production pipeline must parallelize decode/resize/encode.

- [ ] Design REST API for thumbnail ingestion
  - `POST /encode/image` - accepts JPEG, returns embedding + ID
  - `POST /encode/batch` - batch processing endpoint
- [ ] Implement thumbnail preprocessing (resize, normalize)
- [ ] Create embedding storage schema
  - Vector index (FAISS)
  - Metadata (camera_id, timestamp, thumbnail_path, embedding_id)
- [ ] Build thumbnail queue system (handle 960 images/min)
- [ ] Implement error handling and retry logic
- [ ] Add monitoring/logging (encoding time, queue depth)

**Deliverables**:
- Thumbnail encoding service (REST API)
- FAISS vector index with metadata DB
- Performance metrics dashboard

### Phase 3: Search Backend (Week 5-6) — 🔄 IN PROGRESS
**Goal**: Implement semantic search functionality
**Status**: text→image query path works on the PN (`web_pn/pn_app.py`, HF text tower into the TRT image space). Filtering, IVFPQ tuning, and sharded search still to do.

- [ ] Design search API
  - `POST /search/text` - accepts query text, returns top-K results
  - `GET /search/results/{result_id}` - retrieve full result details
- [ ] Implement text query encoding
- [ ] Vector similarity search (FAISS cosine similarity)
- [ ] Result ranking and filtering
  - Time range filtering
  - Camera ID filtering
  - Confidence threshold
- [ ] Thumbnail retrieval and caching
- [ ] Optimize search latency (target: <500ms for 1M embeddings)

**Deliverables**:
- Search API with filtering capabilities
- Optimized vector search (<500ms latency)
- Result caching system

### Phase 4: GN Integration (Week 7-8)
**Goal**: Connect GN and PN systems

- [ ] Create GN→PN client library
  - Thumbnail upload to PN
  - Query submission
  - Result retrieval
- [ ] Implement thumbnail streaming from GN to PN
  - HTTP multipart upload or gRPC streaming
  - Bandwidth optimization (compression if needed)
- [ ] Build thumbnail generation on GN side
  - Extract frame every 2 seconds per camera
  - Resize to 640×360
  - JPEG compression
- [ ] Handle network failures and reconnection
- [ ] End-to-end testing with real camera feeds

**Deliverables**:
- GN-PN communication protocol
- Thumbnail streaming pipeline
- Integration test suite

### Phase 5: Web Interface & User Experience (Week 9-10)
**Goal**: Build user-facing search interface on GN

- [ ] Design search UI
  - Text input for queries
  - Thumbnail grid display
  - Timeline view
  - Camera filter controls
- [ ] Implement real-time search
  - Search as you type (debounced)
  - Loading states
  - Result pagination
- [ ] Add result interactions
  - Click thumbnail → jump to video timestamp
  - Filter by camera, time range
  - Export results
- [ ] Performance optimization (lazy loading, thumbnail caching)
- [ ] Mobile-responsive design (for OSD/tablets)

**Deliverables**:
- Web-based search interface
- Video playback integration
- User documentation

### Phase 6: Optimization & Production Readiness (Week 11-12)
**Goal**: Optimize for scale and prepare for deployment

- [ ] Performance optimization
  - Batch encoding optimization
  - FAISS index tuning (IVF, PQ compression for large datasets)
  - Database query optimization
- [ ] Scalability testing
  - Test with 32 cameras, 24-hour retention
  - Measure storage requirements
  - Stress test search latency
- [ ] Monitoring and observability
  - Prometheus metrics
  - Grafana dashboards
  - Alert rules
- [ ] Backup and recovery
  - Vector index backup strategy
  - Metadata database backup
- [ ] Documentation
  - Deployment guide
  - API documentation
  - Troubleshooting guide

**Deliverables**:
- Production-ready system
- Monitoring dashboards
- Complete documentation

---

## Database Design

> **⚠️ Under active design (2026-06-01).** The schema below is the original sketch. The production direction now being worked out: FAISS holds only vectors + int64 IDs (`IndexIDMap2`), joined to SQLite for metadata; **daily time-shards** with retention by dropping the oldest shard files (FAISS in-place delete is O(N)); **`IndexIVFPQ`** compression once a shard exceeds memory; streaming ingest via an **immutable-history + one writable active shard** pattern (FAISS isn't thread-safe for concurrent add+search). To be finalized in discussion, then this section updated.

### Vector Storage (FAISS)
- **Index Type**: `IndexFlatIP` (inner product, exact search) for <100K embeddings — current demo
- **Index Type**: `IndexIVFPQ` (inverted file + product quantization) for large/sharded scale
- **Embedding Dimension**: **1024** (SigLIP2-L/16 output)
- **Storage Size**: ~4 KB/vector flat fp32 (1024-dim); ~32–64 B/vector under PQ compression

### Metadata Database Schema

```sql
CREATE TABLE thumbnails (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding_id INTEGER NOT NULL,          -- FAISS index ID
    camera_id INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    thumbnail_path TEXT NOT NULL,           -- Path on GN storage
    embedding_norm REAL,                    -- For normalization
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_camera_timestamp (camera_id, timestamp),
    INDEX idx_timestamp (timestamp)
);

CREATE TABLE cameras (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    location TEXT,
    enabled BOOLEAN DEFAULT TRUE
);

CREATE TABLE search_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    num_results INTEGER,
    latency_ms REAL
);
```

---

## API Design

### PN REST API Endpoints

#### Encoding
```
POST /api/v1/encode/image
Content-Type: multipart/form-data
Body: 
  - image: JPEG file
  - camera_id: integer
  - timestamp: ISO 8601 datetime

Response:
{
  "embedding_id": 12345,
  "status": "success",
  "processing_time_ms": 45
}
```

```
POST /api/v1/encode/batch
Content-Type: multipart/form-data
Body: Multiple images with metadata

Response:
{
  "results": [
    {"embedding_id": 12345, "status": "success"},
    ...
  ],
  "total_processing_time_ms": 320
}
```

#### Search
```
POST /api/v1/search
Content-Type: application/json
Body:
{
  "query": "woman in red dress",
  "top_k": 20,
  "filters": {
    "camera_ids": [1, 2, 3],
    "start_time": "2026-05-12T08:00:00Z",
    "end_time": "2026-05-12T18:00:00Z",
    "min_confidence": 0.25
  }
}

Response:
{
  "query": "woman in red dress",
  "results": [
    {
      "embedding_id": 12345,
      "camera_id": 2,
      "timestamp": "2026-05-12T14:23:45Z",
      "thumbnail_path": "/recordings/cam2/2026-05-12/14-23-45.jpg",
      "similarity_score": 0.87
    },
    ...
  ],
  "search_time_ms": 125,
  "total_matches": 18
}
```

#### Health & Monitoring
```
GET /api/v1/health
Response:
{
  "status": "healthy",
  "model_loaded": true,
  "faiss_index_size": 1234567,
  "gpu_memory_used_mb": 3456,
  "uptime_seconds": 86400
}

GET /api/v1/metrics
Response: Prometheus-formatted metrics
```

---

## Performance Targets

### Encoding Performance
- **Single image encoding**: <100ms (TensorRT FP16)
- **Batch encoding (16 images)**: <800ms (~50ms per image)
- **Throughput**: 960 images/min sustained (32 cameras × 30 thumbnails/min)
- **GPU memory usage**: <8GB (leave room for FAISS)

### Search Performance
- **Text encoding**: <50ms
- **Vector search (100K embeddings)**: <100ms
- **Vector search (1M embeddings)**: <500ms
- **End-to-end search latency**: <1s

### Storage Estimates
- **Embeddings**: 1KB × 960/min × 60 min × 24 hours × 7 days = ~9.3GB per week
- **Thumbnails**: 20KB × 960/min × 60 min × 24 hours × 7 days = ~186GB per week
- **Video**: Depends on bitrate and cameras (not stored on PN)

### Scalability Limits (Single PN)
- **Max cameras**: 32 cameras (with 2-second sampling)
- **Max retention**: 30 days (with FAISS compression)
- **Max search corpus**: 10M embeddings (~10GB compressed)

---

## Key Technical Decisions

### Decision 1: Text Encoding Location
**Decision**: Text encoding happens on PN  
**Rationale**: GN lacks GPU, PN already has CLIP model loaded, minimal latency increase  
**Trade-off**: Adds network round-trip, but only for search queries (low frequency)

### Decision 2: Database Location
**Decision**: Vector database on PN (initial implementation)  
**Rationale**: 
- PN has GPU for FAISS-GPU acceleration
- Tight coupling with encoding pipeline
- Lower latency for search
**Future**: Evaluate GN storage if PN runs out of space or need centralized access

### Decision 3: Model Selection — UPDATED 2026-06-01
**Decision**: **SigLIP2-L/16-256** (HuggingFace weights), TensorRT FP16 via torch2trt. 384px variant reserved as a future swap; system is built model-swappable.  
**Rationale**: 
- Full Jetson sweep (EVA-02-B/L, ViT-H, ViT-bigG, SigLIP2-B/L/SO400M) found SigLIP2-L-256 to be the best-accuracy config that clears all throughput targets: **3,145 img/min** at **cos 0.999974** vs FP32, 2.3 GB.
- EVA-02-L/14-336 (the original pick) **cannot reach 960 img/min** (786 max, FP16 TRT); INT8 is a dead end on this board.
- SigLIP2 gives richer embeddings (1024-dim) than EVA-02-L (768) / EVA-02-B (512).
**Original plan**: EVA-02-L/14 (336px) via OpenCLIP — superseded. Full analysis in [docs/JETSON_BENCHMARK_2026.md](./docs/JETSON_BENCHMARK_2026.md).

### Decision 4: Vector Index Type
**Decision**: Start with `IndexFlatIP`, migrate to `IndexIVFPQ` when >100K embeddings  
**Rationale**: 
- Flat index = exact search, simpler to debug
- IVF+PQ = approximate but much faster for large datasets
**Migration**: Rebuild index as offline process

### Decision 5: Communication Protocol
**Decision**: HTTP REST API (JSON) for GN↔PN  
**Rationale**: 
- Simple to implement and debug
- Widely supported
- Sufficient for 960 images/min workload
**Alternative**: gRPC if batching requires streaming or lower latency

---

## Testing Strategy

### Unit Tests
- Image preprocessing pipeline
- Embedding normalization
- FAISS index operations
- Database CRUD operations
- API endpoint validation

### Integration Tests
- End-to-end encoding flow (thumbnail → embedding → storage)
- End-to-end search flow (query → encode → search → results)
- GN-PN communication with network failures
- Database consistency checks

### Performance Tests
- Encoding throughput under load (960 images/min sustained)
- Search latency with varying corpus sizes (1K, 10K, 100K, 1M embeddings)
- GPU memory usage over time
- Disk I/O and storage growth

### User Acceptance Tests
- Search accuracy for common queries
- Response time meets user expectations (<2s end-to-end)
- UI responsiveness
- Edge cases (no results, ambiguous queries)

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| TensorRT optimization fails to meet throughput target | High | Fall back to PyTorch JIT, consider smaller model (EVA-02-B) |
| Jetson Orin Nano runs out of memory | High | Reduce batch size, use INT8 quantization, offload old embeddings |
| Search accuracy not good enough | Medium | Fine-tune model on surveillance data, adjust preprocessing |
| Network bandwidth GN→PN becomes bottleneck | Medium | Compress thumbnails, batch uploads, prioritize cameras |
| FAISS index grows too large for PN storage | Low | Use PQ compression, implement retention policy, move to GN |
| User queries too ambiguous | Low | Add query suggestions, show example searches |

---

## Future Enhancements

### Phase 7+ (Post-MVP)
- **Multi-modal search**: Combine text + reference image ("find more like this")
- **Temporal queries**: "person entering building between 2-3 PM"
- **Object tracking**: Link detections across cameras
- **Alerts**: Notify when specific query matches new thumbnails
- **Model fine-tuning**: Train on domain-specific surveillance data
- **Distributed PN**: Scale to multiple PNs for >32 cameras
- **Mobile app**: iOS/Android app for remote search
- **Activity heatmaps**: Visualize where/when events occur

---

## Success Criteria

### Technical Metrics
- ✅ 960 images/min encoding throughput
- ✅ <1s end-to-end search latency
- ✅ >80% search accuracy (user satisfaction survey)
- ✅ 99.9% uptime for PN service
- ✅ <8GB GPU memory usage

### User Experience
- ✅ Natural language queries work intuitively
- ✅ Results appear within 2 seconds
- ✅ Top 5 results include the target in 90% of cases
- ✅ System handles 32 cameras without degradation

### Business Goals
- ✅ Reduce time to find relevant footage from hours to seconds
- ✅ Increase user adoption of smart search feature
- ✅ Differentiate GN product from competitors

---

## References

- **EVA-02 Paper**: https://arxiv.org/abs/2303.11331
- **OpenCLIP**: https://github.com/mlfoundations/open_clip
- **FAISS**: https://github.com/facebookresearch/faiss
- **Jetson Orin Docs**: https://developer.nvidia.com/embedded/jetson-orin
- **TensorRT**: https://developer.nvidia.com/tensorrt

---

## Appendix

### Hardware Specifications

**Jetson Orin Nano 8GB Dev Kit**:
- GPU: 1024-core NVIDIA Ampere (with Tensor Cores)
- CPU: 6-core Arm Cortex-A78AE
- Memory: 8GB LPDDR5 (or 16GB variant)
- Storage: NVMe SSD recommended
- Power: 7-15W

**Estimated Costs**:
- Jetson Orin Nano 16GB: ~$599
- 256GB NVMe SSD: ~$50
- Power supply: ~$30
- **Total per PN**: ~$680

### Glossary

- **GN**: Gateway/NVR (Network Video Recorder) - records and stores camera feeds
- **PN**: Processing Node - Jetson device that runs CLIP encoding and search
- **CLIP**: Contrastive Language-Image Pretraining - multimodal embedding model
- **EVA-02**: Enhanced Vision Transformer variant with improved CLIP training
- **FAISS**: Facebook AI Similarity Search - vector database for efficient similarity search
- **TensorRT**: NVIDIA inference optimization library for GPU acceleration
- **Embedding**: Vector representation of image or text in high-dimensional space
- **Vector Search**: Finding nearest neighbors in embedding space (similar images/text)
