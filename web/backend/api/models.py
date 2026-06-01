"""Pydantic models for API requests and responses"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# Model info schemas
class ModelInfo(BaseModel):
    """Information about a CLIP model"""
    id: str
    name: str
    display_name: str
    params: str
    embed_dim: int
    loaded: bool = False


class LoRAInfo(BaseModel):
    """Information about a LoRA adapter"""
    id: str
    name: str
    base_model: str
    path: str


class DatasetInfo(BaseModel):
    """Information about an indexed dataset"""
    id: str
    name: str
    num_embeddings: int
    model: str
    index_path: str
    metadata_path: str


# Search request/response schemas
class SearchRequest(BaseModel):
    """Single model search request"""
    query: str = Field(..., min_length=1, description="Search query text")
    model: str = Field(..., description="Model ID (e.g., 'eva02-b')")
    dataset: str = Field(..., description="Dataset ID to search")
    lora: Optional[str] = Field(None, description="Optional LoRA adapter ID")
    top_k: int = Field(20, ge=1, le=100, description="Number of results")


class SearchResult(BaseModel):
    """Single search result"""
    rank: int
    score: float
    image_path: str
    image_url: str
    thumbnail_url: str


class SearchTiming(BaseModel):
    """Search timing breakdown"""
    text_encoding_ms: float
    search_ms: float
    total_ms: float


class SearchResponse(BaseModel):
    """Single model search response"""
    query: str
    model: str
    lora: Optional[str]
    dataset: str
    results: List[SearchResult]
    timing: SearchTiming
    num_results: int


# Comparison schemas
class ComparisonModelConfig(BaseModel):
    """Model configuration for comparison"""
    model: str
    lora: Optional[str] = None


class CompareRequest(BaseModel):
    """Multi-model comparison request"""
    query: str = Field(..., min_length=1, description="Search query text")
    models: List[ComparisonModelConfig] = Field(..., min_items=1, max_items=3)
    dataset: str = Field(..., description="Dataset ID to search")
    top_k: int = Field(20, ge=1, le=100, description="Number of results")


class ComparisonResult(BaseModel):
    """Results for one model in comparison"""
    model: str
    lora: Optional[str]
    results: List[SearchResult]
    timing: SearchTiming


class CompareResponse(BaseModel):
    """Multi-model comparison response"""
    query: str
    dataset: str
    comparisons: List[ComparisonResult]


# List endpoints responses
class ModelsListResponse(BaseModel):
    """List of available models and adapters"""
    base_models: List[ModelInfo]
    lora_adapters: List[LoRAInfo]


class DatasetsListResponse(BaseModel):
    """List of available datasets"""
    datasets: List[DatasetInfo]


# Health check
class HealthResponse(BaseModel):
    """API health status"""
    status: str
    device: str
    gpu_name: Optional[str] = None
    models_loaded: List[str]
    datasets_available: List[str]
