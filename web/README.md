# CLIP Search Web Interface

Interactive web interface for comparing CLIP models and testing semantic search queries.

## Features

- **Model Selection**: Choose from EVA-02-B, EVA-02-L, SigLIP 2, and other models
- **Side-by-Side Comparison**: Compare 2-3 models with same query simultaneously
- **LoRA Adapter Support**: Load and test fine-tuned adapters
- **Interactive Search**: Real-time semantic search with visual results
- **Result Analysis**: View similarity scores, click to enlarge, export results

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Frontend (HTML + JavaScript)                   │
│  - Search input box                             │
│  - Model selection dropdowns                    │
│  - Split-screen comparison view                 │
│  - Image grid with scores                       │
└────────────┬────────────────────────────────────┘
             │ HTTP REST API
             ↓
┌─────────────────────────────────────────────────┐
│  Backend (FastAPI)                              │
│  - /api/models - List available models          │
│  - /api/search - Perform semantic search        │
│  - /api/compare - Multi-model comparison        │
│  - /api/embeddings - List indexed datasets      │
└────────────┬────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────┐
│  Core Services                                  │
│  - ModelManager: Load/cache CLIP models         │
│  - EmbeddingManager: Load FAISS indices         │
│  - SearchService: Execute queries               │
└─────────────────────────────────────────────────┘
```

## Tech Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Uvicorn**: ASGI server
- **CLIP models**: EVA-02-B/L, SigLIP 2, etc.
- **FAISS**: Vector search
- **PEFT**: LoRA adapter loading

### Frontend
- **HTML5 + CSS3**: Responsive layout
- **Vanilla JavaScript**: No framework needed for MVP
- **Fetch API**: Async HTTP requests
- **CSS Grid**: Side-by-side comparison layout

## Project Structure

```
web/
├── README.md                # This file
├── backend/
│   ├── main.py             # FastAPI application
│   ├── api/
│   │   ├── routes.py       # API endpoints
│   │   └── models.py       # Pydantic schemas
│   ├── services/
│   │   ├── model_manager.py    # Model loading/caching
│   │   ├── embedding_manager.py # FAISS index management
│   │   └── search_service.py   # Search logic
│   └── config.py           # Configuration
├── frontend/
│   ├── index.html          # Main page
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css   # Styles
│   │   └── js/
│   │       └── app.js      # Frontend logic
│   └── assets/             # Images, icons
└── requirements.txt        # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
cd web/
pip install -r requirements.txt
```

### 2. Configure

Edit `backend/config.py`:
```python
MODELS_CONFIG = {
    "eva02-b": {
        "name": "EVA02-B-16",
        "pretrained": "merged2b_s8b_b131k"
    },
    "eva02-l": {
        "name": "EVA02-L-14-336",
        "pretrained": "merged2b_s6b_b61k"
    }
}

EMBEDDINGS_DIR = "/path/to/embeddings"
```

### 3. Run Server

```bash
cd backend/
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access UI

Open browser: http://localhost:8000

## API Endpoints

### GET /api/models
List available CLIP models and LoRA adapters.

**Response:**
```json
{
  "base_models": [
    {
      "id": "eva02-b",
      "name": "EVA-02-B/16",
      "params": "150M",
      "loaded": true
    }
  ],
  "lora_adapters": [
    {
      "id": "vehicle-lora",
      "name": "Vehicle Adapter",
      "base_model": "eva02-b"
    }
  ]
}
```

### GET /api/embeddings
List indexed datasets (FAISS indices).

**Response:**
```json
{
  "datasets": [
    {
      "id": "crowdhuman",
      "name": "CrowdHuman Train",
      "num_embeddings": 15929,
      "model": "eva02-b"
    }
  ]
}
```

### POST /api/search
Search single model.

**Request:**
```json
{
  "query": "person in red jacket",
  "model": "eva02-b",
  "dataset": "crowdhuman",
  "lora": null,
  "top_k": 20
}
```

**Response:**
```json
{
  "query": "person in red jacket",
  "model": "eva02-b",
  "results": [
    {
      "rank": 1,
      "score": 0.8532,
      "image_path": "/path/to/image.jpg",
      "image_url": "/api/image/12345"
    }
  ],
  "timing": {
    "text_encoding_ms": 0.7,
    "search_ms": 12.4,
    "total_ms": 13.1
  }
}
```

### POST /api/compare
Compare multiple models side-by-side.

**Request:**
```json
{
  "query": "blue Tesla Model 3",
  "models": [
    {
      "model": "eva02-b",
      "lora": null
    },
    {
      "model": "eva02-l",
      "lora": "vehicle-lora"
    }
  ],
  "dataset": "coco-vehicles",
  "top_k": 20
}
```

**Response:**
```json
{
  "query": "blue Tesla Model 3",
  "comparisons": [
    {
      "model": "eva02-b",
      "lora": null,
      "results": [...],
      "timing": {...}
    },
    {
      "model": "eva02-l",
      "lora": "vehicle-lora",
      "results": [...],
      "timing": {...}
    }
  ]
}
```

### GET /api/image/{image_id}
Serve image file for display.

## Usage Examples

### Single Model Search
1. Select model: EVA-02-B
2. Enter query: "person wearing red"
3. Click "Search"
4. View results with scores

### Side-by-Side Comparison
1. Select Model 1: EVA-02-B (base)
2. Select Model 2: EVA-02-L (base)
3. Enter query: "blue Tesla Model 3"
4. Click "Compare"
5. View results side-by-side

### LoRA Adapter Testing
1. Select Model: EVA-02-B
2. Select Adapter: vehicle-lora
3. Enter query: "red Toyota Camry"
4. Compare with base model

## Development Notes

### Model Caching
Models are loaded once and cached in memory. GPU memory usage:
- EVA-02-B: ~1.5 GB
- EVA-02-L: ~3.5 GB
- LoRA adapters: +10 MB each

### Performance
- Text encoding: <1ms (cached after first query)
- FAISS search: 10-50ms (depends on index size)
- Image loading: 20-100ms (network I/O)
- Total latency: <200ms typical

### Scalability
- Current: Single GPU, 2-3 models loaded
- Future: Model on-demand loading, multi-GPU support

## Configuration

### backend/config.py

```python
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
LORA_DIR = PROJECT_ROOT / "lora_adapters"

# Model configurations
MODELS_CONFIG = {
    "eva02-b": {
        "name": "EVA02-B-16",
        "pretrained": "merged2b_s8b_b131k",
        "embed_dim": 512
    },
    "eva02-l": {
        "name": "EVA02-L-14-336",
        "pretrained": "merged2b_s6b_b61k",
        "embed_dim": 768
    }
}

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
CORS_ORIGINS = ["*"]  # Configure for production

# Search settings
DEFAULT_TOP_K = 20
MAX_TOP_K = 100

# GPU settings
DEVICE = "cuda" if os.path.exists("/dev/nvidia0") else "cpu"
```

## Next Steps

### MVP (Week 1)
- [x] Project structure
- [ ] FastAPI backend with /api/search endpoint
- [ ] Model manager (load EVA-02-B)
- [ ] Embedding manager (load FAISS indices)
- [ ] Simple HTML frontend
- [ ] Single model search

### Phase 2 (Week 2)
- [ ] /api/compare endpoint
- [ ] Side-by-side comparison UI
- [ ] Model selection dropdown
- [ ] Result visualization improvements

### Phase 3 (Week 3)
- [ ] LoRA adapter loading
- [ ] Adapter management UI
- [ ] Performance metrics display
- [ ] Export results feature

### Future Enhancements
- [ ] Query history
- [ ] Batch search
- [ ] Advanced filters (date, camera, score threshold)
- [ ] User accounts and saved searches
- [ ] Mobile responsive design
- [ ] Dark mode

## Troubleshooting

### Port already in use
```bash
lsof -i :8000
kill -9 <PID>
```

### CORS errors
Update `CORS_ORIGINS` in `backend/config.py`

### Model not loading
Check CUDA availability:
```python
import torch
print(torch.cuda.is_available())
```

### Slow search
- Check FAISS index type (use IndexIVFPQ for large datasets)
- Verify GPU is being used for search
- Consider reducing embedding dataset size

## Contact

Last Updated: 2026-05-12
