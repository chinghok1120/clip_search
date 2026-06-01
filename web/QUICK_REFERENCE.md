# Quick Reference Card

## Start Server
```bash
cd ~/projects/clip_search/web
./start.sh
```
Open: http://localhost:8000/

## Stop Server
`Ctrl+C` in terminal

## Files Created (17 files, ~800 lines)
```
web/
├── backend/          # FastAPI backend (10 Python files)
├── frontend/         # Web UI (1 HTML, 1 CSS, 1 JS)
├── README.md         # Full docs
├── QUICKSTART.md     # Setup guide
└── start.sh          # Startup script
```

## Dataset Ready
- **crowdhuman.faiss** - 15,929 images indexed
- Location: `~/projects/clip_search/embeddings/`

## Example Queries
- "person in red jacket"
- "person with backpack"
- "people walking"

## API Endpoints
- GET  `/api/health` - Status check
- GET  `/api/models` - List models
- GET  `/api/datasets` - List datasets
- POST `/api/search` - Search (single model)
- POST `/api/compare` - Compare models

Full API docs: http://localhost:8000/docs

## Next Session
1. Test the interface
2. Discuss layout changes
3. Decide on features to add
4. Consider LoRA training

## Status
✅ MVP complete - ready to test!
