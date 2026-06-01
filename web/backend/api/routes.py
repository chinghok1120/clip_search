"""API routes for CLIP search interface"""
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, FileResponse
from PIL import Image

from .models import (
    SearchRequest, SearchResponse, SearchResult, SearchTiming,
    CompareRequest, CompareResponse, ComparisonResult,
    ModelsListResponse, ModelInfo, LoRAInfo,
    DatasetsListResponse, DatasetInfo,
    HealthResponse
)
from ..services.model_manager import model_manager
from ..services.embedding_manager import embedding_manager
from ..services.search_service import search_service
from ..config import MAX_IMAGE_SIZE, IMAGE_QUALITY, DEVICE

router = APIRouter()


def _validate_model_dataset_compat(model_id: str, dataset_id: str):
    """Raise 400 if model embed_dim doesn't match dataset's stored embed_dim"""
    from ..config import MODELS_CONFIG
    dataset_info = embedding_manager.datasets_info.get(dataset_id)
    if not dataset_info:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")

    dataset_model = dataset_info.get("model", "unknown")
    if dataset_model == "unknown":
        return  # Can't validate, allow through

    model_config = MODELS_CONFIG.get(model_id)
    if not model_config:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    expected_config = MODELS_CONFIG.get(dataset_model)
    if expected_config and model_config["embed_dim"] != expected_config["embed_dim"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model_id}' produces {model_config['embed_dim']}-dim embeddings, "
                f"but dataset '{dataset_id}' was encoded with '{dataset_model}' "
                f"({expected_config['embed_dim']}-dim). "
                f"Please select '{dataset_model}' or re-encode the dataset with '{model_id}'."
            )
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and GPU status"""
    gpu_name = None
    if DEVICE == "cuda":
        import torch
        gpu_name = torch.cuda.get_device_name(0)

    return HealthResponse(
        status="healthy",
        device=DEVICE,
        gpu_name=gpu_name,
        models_loaded=list(model_manager.loaded_models.keys()),
        datasets_available=[d["id"] for d in embedding_manager.get_available_datasets()]
    )


@router.get("/models", response_model=ModelsListResponse)
async def list_models():
    """List available CLIP models and LoRA adapters"""
    base_models = [
        ModelInfo(**model_info)
        for model_info in model_manager.get_available_models()
    ]

    lora_adapters = [
        LoRAInfo(**adapter_info)
        for adapter_info in model_manager.get_available_lora_adapters()
    ]

    return ModelsListResponse(
        base_models=base_models,
        lora_adapters=lora_adapters
    )


@router.get("/datasets", response_model=DatasetsListResponse)
async def list_datasets():
    """List available indexed datasets"""
    datasets = [
        DatasetInfo(**dataset_info)
        for dataset_info in embedding_manager.get_available_datasets()
    ]

    return DatasetsListResponse(datasets=datasets)


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform semantic search with single model"""
    try:
        # Validate model is compatible with dataset
        _validate_model_dataset_compat(request.model, request.dataset)

        # Perform search
        result = search_service.search(
            query=request.query,
            model_id=request.model,
            dataset_id=request.dataset,
            lora_id=request.lora,
            top_k=request.top_k
        )

        # Format results with image URLs
        formatted_results = []
        for res in result["results"]:
            formatted_results.append(SearchResult(
                rank=res["rank"],
                score=res["score"],
                image_path=res["image_path"],
                image_url=f"/api/image/{res['image_id']}?dataset={request.dataset}",
                thumbnail_url=f"/api/image/{res['image_id']}?dataset={request.dataset}&size=thumb"
            ))

        return SearchResponse(
            query=request.query,
            model=request.model,
            lora=request.lora,
            dataset=request.dataset,
            results=formatted_results,
            timing=SearchTiming(**result["timing"]),
            num_results=result["num_results"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare", response_model=CompareResponse)
async def compare(request: CompareRequest):
    """Compare multiple models side-by-side"""
    try:
        # Validate all models are compatible with dataset
        for mc in request.models:
            _validate_model_dataset_compat(mc.model, request.dataset)

        # Perform comparison
        comparisons = search_service.compare(
            query=request.query,
            model_configs=[
                {"model": mc.model, "lora": mc.lora}
                for mc in request.models
            ],
            dataset_id=request.dataset,
            top_k=request.top_k
        )

        # Format results
        formatted_comparisons = []
        for comp in comparisons:
            formatted_results = []
            for res in comp["results"]:
                formatted_results.append(SearchResult(
                    rank=res["rank"],
                    score=res["score"],
                    image_path=res["image_path"],
                    image_url=f"/api/image/{res['image_id']}?dataset={request.dataset}",
                    thumbnail_url=f"/api/image/{res['image_id']}?dataset={request.dataset}&size=thumb"
                ))

            formatted_comparisons.append(ComparisonResult(
                model=comp["model"],
                lora=comp["lora"],
                results=formatted_results,
                timing=SearchTiming(**comp["timing"])
            ))

        return CompareResponse(
            query=request.query,
            dataset=request.dataset,
            comparisons=formatted_comparisons
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/image/{image_id}")
async def get_image(
    image_id: str,
    dataset: str = Query(..., description="Dataset ID"),
    size: Optional[str] = Query(None, description="'thumb' for thumbnail")
):
    """Serve image file"""
    try:
        # Get metadata
        metadata_list = embedding_manager.get_metadata(dataset, [int(image_id)])
        if not metadata_list or "error" in metadata_list[0]:
            raise HTTPException(status_code=404, detail="Image not found")

        image_path = Path(metadata_list[0]["path"])

        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image file not found")

        # If thumbnail requested, resize
        if size == "thumb":
            img = Image.open(image_path)
            img.thumbnail((256, 256))

            # Convert to bytes
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=IMAGE_QUALITY)
            buffer.seek(0)

            return Response(content=buffer.getvalue(), media_type="image/jpeg")

        # Otherwise serve full image
        return FileResponse(image_path)

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image ID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "models": model_manager.get_memory_usage(),
        "datasets": embedding_manager.get_stats()
    }
