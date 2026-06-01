"""Configuration for CLIP Search Web Interface"""
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
LORA_DIR = PROJECT_ROOT / "lora_adapters"
DATASETS_DIR = Path.home() / "datasets"

# Model configurations
MODELS_CONFIG = {
    "eva02-b": {
        "name": "EVA02-B-16",
        "pretrained": "merged2b_s8b_b131k",
        "embed_dim": 512,
        "file_suffix": "eva02-b-16",
        "display_name": "EVA-02-B/16 (150M params)"
    },
    "eva02-l": {
        "name": "EVA02-L-14-336",
        "pretrained": "merged2b_s6b_b61k",
        "embed_dim": 768,
        "file_suffix": "eva02-l-14-336",
        "display_name": "EVA-02-L/14 (428M params)"
    },
    "dfn5b-h": {
        "name": "ViT-H-14-378-quickgelu",
        "pretrained": "dfn5b",
        "embed_dim": 1024,
        "file_suffix": "vit-h-14-378-quickgelu",
        "display_name": "DFN5B ViT-H/14-378 (~1B params)"
    },
    "bigg": {
        "name": "ViT-bigG-14",
        "pretrained": "laion2b_s39b_b160k",
        "embed_dim": 1280,
        "file_suffix": "vit-bigg-14",
        "display_name": "LAION ViT-bigG/14 (2.5B params)"
    },
    "siglip2-l": {
        "name": "ViT-L-16-SigLIP2-256",
        "pretrained": "webli",
        "embed_dim": 1024,
        "file_suffix": "vit-l-16-siglip2-256",
        "display_name": "SigLIP2 L/16-256 (~300M params)"
    },
    "siglip2-l-384": {
        "name": "ViT-L-16-SigLIP2-384",
        "pretrained": "webli",
        "embed_dim": 1024,
        "file_suffix": "vit-l-16-siglip2-384",
        "display_name": "SigLIP2 L/16-384 (~300M params)"
    },
    "siglip2-so400m": {
        "name": "ViT-SO400M-14-SigLIP2-378",
        "pretrained": "webli",
        "embed_dim": 1152,
        "file_suffix": "vit-so400m-14-siglip2-378",
        "display_name": "SigLIP2 SO400M/14-378 (~400M params)"
    }
}

# Reverse map: file_suffix → model_id (used by embedding_manager to parse filenames)
SUFFIX_TO_MODEL_ID = {v["file_suffix"]: k for k, v in MODELS_CONFIG.items()}

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
CORS_ORIGINS = ["*"]  # Allow all origins for development

# Search settings
DEFAULT_TOP_K = 20
MAX_TOP_K = 100

# GPU settings
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Image serving
MAX_IMAGE_SIZE = (800, 800)  # Max dimensions for serving images
IMAGE_QUALITY = 85  # JPEG quality
