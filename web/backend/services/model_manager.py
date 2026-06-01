"""Model management service for loading and caching CLIP models"""
import time
from typing import Dict, Optional, Tuple
import torch
import open_clip
from pathlib import Path

from ..config import MODELS_CONFIG, DEVICE, LORA_DIR


class ModelManager:
    """Manages CLIP model loading, caching, and LoRA adapters"""

    def __init__(self):
        self.device = DEVICE
        self.loaded_models: Dict[str, Tuple] = {}  # model_id -> (model, preprocess, tokenizer)
        self.lora_adapters: Dict[str, Path] = {}  # Discover LoRA adapters

        # Scan for LoRA adapters
        self._scan_lora_adapters()

    def _scan_lora_adapters(self):
        """Scan LORA_DIR for available adapters"""
        if LORA_DIR.exists():
            for adapter_dir in LORA_DIR.iterdir():
                if adapter_dir.is_dir():
                    # Check if it has adapter files
                    if (adapter_dir / "adapter_config.json").exists():
                        self.lora_adapters[adapter_dir.name] = adapter_dir

    def get_available_models(self) -> list:
        """Get list of available base models"""
        models = []
        for model_id, config in MODELS_CONFIG.items():
            models.append({
                "id": model_id,
                "name": config["name"],
                "display_name": config["display_name"],
                "params": config["display_name"].split("(")[1].split(")")[0] if "(" in config["display_name"] else "Unknown",
                "embed_dim": config["embed_dim"],
                "loaded": model_id in self.loaded_models
            })
        return models

    def get_available_lora_adapters(self) -> list:
        """Get list of available LoRA adapters"""
        adapters = []
        for adapter_id, adapter_path in self.lora_adapters.items():
            # Try to read base model from config
            base_model = "unknown"
            config_file = adapter_path / "adapter_config.json"
            if config_file.exists():
                import json
                with open(config_file) as f:
                    adapter_config = json.load(f)
                    # This is a placeholder - actual format depends on PEFT
                    base_model = adapter_config.get("base_model_name_or_path", "unknown")

            adapters.append({
                "id": adapter_id,
                "name": adapter_id.replace("_", " ").title(),
                "base_model": base_model,
                "path": str(adapter_path)
            })
        return adapters

    def load_model(self, model_id: str) -> Tuple:
        """Load a CLIP model if not already loaded

        Args:
            model_id: Model identifier (e.g., 'eva02-b')

        Returns:
            (model, preprocess, tokenizer) tuple
        """
        # Check if already loaded
        if model_id in self.loaded_models:
            return self.loaded_models[model_id]

        # Get model config
        if model_id not in MODELS_CONFIG:
            raise ValueError(f"Unknown model ID: {model_id}")

        config = MODELS_CONFIG[model_id]

        print(f"Loading model {model_id} ({config['display_name']})...")
        start_time = time.time()

        # Load model
        model, _, preprocess = open_clip.create_model_and_transforms(
            config["name"],
            pretrained=config["pretrained"],
            device=self.device
        )

        # Get tokenizer
        tokenizer = open_clip.get_tokenizer(config["name"])

        # Set to eval mode
        model.eval()

        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f}s")

        # Cache
        self.loaded_models[model_id] = (model, preprocess, tokenizer)

        return model, preprocess, tokenizer

    def load_model_with_lora(self, model_id: str, lora_id: str) -> Tuple:
        """Load a CLIP model with LoRA adapter

        Args:
            model_id: Base model identifier
            lora_id: LoRA adapter identifier

        Returns:
            (model, preprocess, tokenizer) tuple
        """
        # For now, load base model (LoRA loading to be implemented)
        # TODO: Implement PEFT LoRA loading
        print(f"LoRA loading not yet implemented, using base model {model_id}")
        return self.load_model(model_id)

    def encode_text(self, model_id: str, query: str, lora_id: Optional[str] = None) -> torch.Tensor:
        """Encode text query using specified model

        Args:
            model_id: Model identifier
            query: Text query string
            lora_id: Optional LoRA adapter ID

        Returns:
            Text embedding tensor (normalized)
        """
        # Load model
        if lora_id:
            model, _, tokenizer = self.load_model_with_lora(model_id, lora_id)
        else:
            model, _, tokenizer = self.load_model(model_id)

        # Tokenize
        text = tokenizer([query])

        # Encode
        with torch.no_grad():
            text_features = model.encode_text(text.to(self.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_embed_dim(self, model_id: str) -> int:
        """Get embedding dimension for a model"""
        if model_id not in MODELS_CONFIG:
            raise ValueError(f"Unknown model ID: {model_id}")
        return MODELS_CONFIG[model_id]["embed_dim"]

    def unload_model(self, model_id: str):
        """Unload a model from memory"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            if self.device == "cuda":
                torch.cuda.empty_cache()
            print(f"✓ Model {model_id} unloaded")

    def get_memory_usage(self) -> dict:
        """Get GPU memory usage statistics"""
        if self.device == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "device": torch.cuda.get_device_name(0)
            }
        return {"device": "cpu"}


# Global instance
model_manager = ModelManager()
