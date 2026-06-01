"""Search service that coordinates models and embeddings"""
import time
from typing import List, Dict, Optional
import numpy as np
import torch

from .model_manager import model_manager
from .embedding_manager import embedding_manager


class SearchService:
    """High-level search service"""

    def __init__(self):
        self.model_manager = model_manager
        self.embedding_manager = embedding_manager

    def search(
        self,
        query: str,
        model_id: str,
        dataset_id: str,
        lora_id: Optional[str] = None,
        top_k: int = 20
    ) -> Dict:
        """Perform semantic search

        Args:
            query: Text query
            model_id: Model to use for encoding
            dataset_id: Dataset to search
            lora_id: Optional LoRA adapter
            top_k: Number of results

        Returns:
            Search results dict with timing info
        """
        timing = {}

        # 1. Encode text query
        text_start = time.time()
        text_features = self.model_manager.encode_text(model_id, query, lora_id)
        text_embedding = text_features.cpu().numpy()
        timing["text_encoding_ms"] = (time.time() - text_start) * 1000

        # 2. Search FAISS index
        search_start = time.time()
        scores, indices = self.embedding_manager.search(dataset_id, text_embedding, top_k)
        timing["search_ms"] = (time.time() - search_start) * 1000

        # 3. Get metadata
        metadata_list = self.embedding_manager.get_metadata(dataset_id, indices.tolist())

        # 4. Format results
        results = []
        for rank, (score, idx, meta) in enumerate(zip(scores, indices, metadata_list), start=1):
            if "error" not in meta:
                results.append({
                    "rank": rank,
                    "score": float(score),
                    "image_path": meta.get("path", ""),
                    "image_id": str(idx)
                })

        timing["total_ms"] = timing["text_encoding_ms"] + timing["search_ms"]

        return {
            "results": results,
            "timing": timing,
            "num_results": len(results)
        }

    def compare(
        self,
        query: str,
        model_configs: List[Dict],
        dataset_id: str,
        top_k: int = 20
    ) -> List[Dict]:
        """Compare multiple models side-by-side

        Args:
            query: Text query
            model_configs: List of {"model": str, "lora": Optional[str]}
            dataset_id: Dataset to search
            top_k: Number of results

        Returns:
            List of search results for each model
        """
        comparisons = []

        for config in model_configs:
            model_id = config["model"]
            lora_id = config.get("lora")

            # Perform search
            result = self.search(query, model_id, dataset_id, lora_id, top_k)

            comparisons.append({
                "model": model_id,
                "lora": lora_id,
                "results": result["results"],
                "timing": result["timing"]
            })

        return comparisons


# Global instance
search_service = SearchService()
