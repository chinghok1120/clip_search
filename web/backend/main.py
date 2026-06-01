"""Main FastAPI application for CLIP Search Web Interface"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pathlib import Path

from .api.routes import router as api_router
from .config import CORS_ORIGINS

# Create FastAPI app
app = FastAPI(
    title="CLIP Search API",
    description="Semantic search interface for CLIP models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Mount frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")

    @app.get("/")
    async def serve_frontend():
        """Serve frontend HTML with no-cache headers"""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(
                index_file,
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
            )
        return {"message": "Frontend not found. Build frontend first."}
else:
    @app.get("/")
    async def root():
        return {
            "message": "CLIP Search API",
            "docs": "/docs",
            "health": "/api/health"
        }


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("=" * 80)
    print("CLIP Search Web Interface Starting...")
    print("=" * 80)
    print(f"\nBackend: FastAPI")
    print(f"API Docs: http://localhost:8000/docs")
    print(f"Frontend: http://localhost:8000/")
    print("\nInitializing services...")

    from .services.model_manager import model_manager
    from .services.embedding_manager import embedding_manager

    # Print available models
    models = model_manager.get_available_models()
    print(f"\n✓ Found {len(models)} base models:")
    for model in models:
        print(f"  - {model['display_name']}")

    # Print available LoRA adapters
    adapters = model_manager.get_available_lora_adapters()
    if adapters:
        print(f"\n✓ Found {len(adapters)} LoRA adapters:")
        for adapter in adapters:
            print(f"  - {adapter['name']}")
    else:
        print(f"\n  No LoRA adapters found in {model_manager.lora_adapters}")

    # Print available datasets
    datasets = embedding_manager.get_available_datasets()
    print(f"\n✓ Found {len(datasets)} indexed datasets:")
    for dataset in datasets:
        print(f"  - {dataset['name']}: {dataset['num_embeddings']} embeddings")

    # Print device info
    from .config import DEVICE
    if DEVICE == "cuda":
        import torch
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n✓ GPU: {gpu_name}")
    else:
        print(f"\n  Device: CPU (GPU not available)")

    print("\n" + "=" * 80)
    print("✓ Ready to serve requests!")
    print("=" * 80 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\nShutting down...")


if __name__ == "__main__":
    import uvicorn
    from .config import API_HOST, API_PORT

    uvicorn.run(
        "backend.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )
