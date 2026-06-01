# Quick Start Guide - CLIP Search Web Interface

Get the web interface running in 5 minutes!

## Prerequisites

- Python virtual environment activated (`venv/`)
- FAISS embeddings generated (at least one dataset)
- CUDA GPU (optional but recommended)

## Step 1: Install Dependencies

```bash
cd web/
pip install -r requirements.txt
```

## Step 2: Verify Embeddings

Ensure you have at least one dataset indexed:

```bash
ls -lh ../embeddings/
# Should show .faiss and .json files
```

If empty, generate embeddings first:

```bash
cd ../scripts/
./encode_images.py --input ~/datasets/crowdhuman/train/images/ --output ../embeddings/crowdhuman
```

## Step 3: Start Server

### Option A: Using start script (recommended)

```bash
cd web/
./start.sh
```

### Option B: Manual start

```bash
cd web/backend/
source ../../venv/bin/activate
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Step 4: Access Interface

Open browser:
- **Web UI**: http://localhost:8000/
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## Usage

### Single Model Search

1. Select **Single Model** mode
2. Choose dataset (e.g., "CrowdHuman")
3. Select model (e.g., "EVA-02-B/16")
4. Enter query: "person in red jacket"
5. Click **Search**

### Compare Models

1. Select **Compare Models** mode
2. Choose dataset
3. Select Model 1: EVA-02-B/16
4. Select Model 2: EVA-02-L/14 (if available)
5. Enter query: "blue Tesla Model 3"
6. Click **Compare**

View results side-by-side to see which model performs better!

### With LoRA Adapters

1. Select model (e.g., EVA-02-B)
2. Choose LoRA adapter (e.g., "Vehicle Adapter")
3. Search and compare with base model

## Example Queries

For **CrowdHuman** dataset:
- "person in red jacket"
- "person with backpack"
- "people walking"
- "man in suit"
- "woman with umbrella"

For **Vehicle** datasets (when available):
- "blue Tesla Model 3"
- "red Toyota Camry"
- "white BMW X5"
- "person next to car"
- "damaged vehicle"

## Troubleshooting

### Port 8000 already in use

```bash
# Find and kill process
lsof -i :8000
kill -9 <PID>
```

### No datasets found

Generate embeddings first:

```bash
cd scripts/
./encode_images.py --input <image_folder> --output ../embeddings/<dataset_name>
```

### CUDA out of memory

Try loading one model at a time, or use CPU mode by editing `backend/config.py`:

```python
DEVICE = "cpu"  # Force CPU mode
```

### Images not loading

Check image paths in metadata.json are absolute and exist:

```bash
head -20 ../embeddings/crowdhuman.json
```

## API Examples

### cURL

```bash
# Health check
curl http://localhost:8000/api/health

# List models
curl http://localhost:8000/api/models

# Search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "person in red jacket",
    "model": "eva02-b",
    "dataset": "crowdhuman",
    "top_k": 20
  }'
```

### Python

```python
import requests

# Search
response = requests.post(
    "http://localhost:8000/api/search",
    json={
        "query": "person in red jacket",
        "model": "eva02-b",
        "dataset": "crowdhuman",
        "top_k": 20
    }
)

results = response.json()
print(f"Found {results['num_results']} results in {results['timing']['total_ms']:.1f}ms")
```

## Performance

Expected latency (RTX 3090):
- Text encoding: <1ms
- FAISS search: 10-50ms
- Total: <100ms

## Next Steps

1. **Generate more datasets**: Encode your surveillance footage
2. **Train LoRA adapters**: Fine-tune on vehicle/person datasets
3. **Compare models**: Test EVA-02-B vs EVA-02-L
4. **Optimize**: Benchmark different batch sizes

## Support

See full documentation: [web/README.md](./README.md)

Issues? Check logs or contact: chinghokuk@gmail.com
