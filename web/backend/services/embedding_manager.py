"""Embedding and FAISS index management service"""
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import faiss

from ..config import EMBEDDINGS_DIR, DEVICE, SUFFIX_TO_MODEL_ID, MODELS_CONFIG


class EmbeddingManager:
    """Manages FAISS indices and metadata for different datasets"""

    def __init__(self):
        self.indices: Dict[str, faiss.Index] = {}  # dataset_id -> FAISS index
        self.metadata: Dict[str, List[dict]] = {}  # dataset_id -> metadata list
        self.datasets_info: Dict[str, dict] = {}  # dataset_id -> info

        # Scan for available datasets
        self._scan_datasets()

    def _scan_datasets(self):
        """Scan EMBEDDINGS_DIR for available FAISS indices"""
        if not EMBEDDINGS_DIR.exists():
            print(f"Warning: Embeddings directory not found: {EMBEDDINGS_DIR}")
            return

        # Look for .faiss files
        for faiss_file in EMBEDDINGS_DIR.rglob("*.faiss"):
            # Corresponding metadata file
            metadata_file = faiss_file.with_suffix(".json")

            if not metadata_file.exists():
                print(f"Warning: Metadata file not found for {faiss_file.name}")
                continue

            # Parse dataset name and model from filename
            # Convention: {dataset_name}_{model_suffix}.faiss
            # e.g. crowdhuman_eva02-b-16.faiss → name=crowdhuman, model=eva02-b
            dataset_name, model_id = self._parse_filename(faiss_file.stem)
            dataset_id = faiss_file.stem  # full stem is the unique ID

            # Load metadata for counts only (don't cache yet)
            with open(metadata_file) as f:
                metadata = json.load(f)

            # Fall back to metadata embed_dim if model not in filename
            if model_id == "unknown":
                model_id = self._detect_model_from_metadata(metadata)

            self.datasets_info[dataset_id] = {
                "id": dataset_id,
                "name": self._make_display_name(dataset_name, model_id),
                "num_embeddings": metadata.get("num_embeddings", len(metadata.get("images", []))),
                "model": model_id,
                "index_path": str(faiss_file),
                "metadata_path": str(metadata_file)
            }

    def _parse_filename(self, stem: str):
        """Parse 'crowdhuman_eva02-b-16' → ('crowdhuman', 'eva02-b')"""
        for suffix, model_id in SUFFIX_TO_MODEL_ID.items():
            if stem.endswith(f"_{suffix}"):
                dataset_name = stem[: -len(f"_{suffix}")]
                return dataset_name, model_id
        return stem, "unknown"

    def _make_display_name(self, dataset_name: str, model_id: str) -> str:
        """Format display name: 'crowdhuman' + 'eva02-b' → 'Crowdhuman (EVA-02-B/16)'"""
        pretty_dataset = dataset_name.replace("_", " ").title()
        model_display = MODELS_CONFIG.get(model_id, {}).get("display_name", model_id)
        # Shorten: "EVA-02-B/16 (150M params)" → "EVA-02-B/16"
        model_short = model_display.split(" (")[0]
        return f"{pretty_dataset} ({model_short})"

    def _detect_model_from_metadata(self, metadata: dict) -> str:
        """Fall back: detect model from embedding_dim stored in metadata"""
        dim = metadata.get("embedding_dim")
        if dim == 512:
            return "eva02-b"
        elif dim == 768:
            return "eva02-l"
        return "unknown"

    def get_available_datasets(self) -> List[dict]:
        """Get list of available datasets"""
        return list(self.datasets_info.values())

    def load_dataset(self, dataset_id: str) -> Tuple[faiss.Index, List[dict]]:
        """Load FAISS index and metadata for a dataset

        Args:
            dataset_id: Dataset identifier

        Returns:
            (faiss_index, metadata_list) tuple
        """
        # Check if already loaded
        if dataset_id in self.indices:
            return self.indices[dataset_id], self.metadata[dataset_id]

        # Get dataset info
        if dataset_id not in self.datasets_info:
            raise ValueError(f"Unknown dataset ID: {dataset_id}")

        info = self.datasets_info[dataset_id]

        print(f"Loading dataset {dataset_id} ({info['name']})...")
        start_time = time.time()

        # Load FAISS index
        index = faiss.read_index(info["index_path"])

        # Move to GPU if available
        if DEVICE == "cuda" and hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print(f"  ✓ Index moved to GPU")
            except Exception as e:
                print(f"  Warning: Could not move index to GPU: {e}")

        # Load metadata - extract the images list from the JSON structure
        with open(info["metadata_path"]) as f:
            raw = json.load(f)
        images = raw.get("images", raw) if isinstance(raw, dict) else raw

        load_time = time.time() - start_time
        print(f"✓ Dataset loaded in {load_time:.2f}s ({len(images)} embeddings)")

        # Cache
        self.indices[dataset_id] = index
        self.metadata[dataset_id] = images

        return index, images

    def search(self, dataset_id: str, query_embedding: np.ndarray, top_k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Search FAISS index for similar embeddings

        Args:
            dataset_id: Dataset to search
            query_embedding: Query embedding vector (normalized)
            top_k: Number of results to return

        Returns:
            (scores, indices) arrays
        """
        # Load dataset if needed
        index, _ = self.load_dataset(dataset_id)

        # Ensure query is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search
        scores, indices = index.search(query_embedding, top_k)

        return scores[0], indices[0]

    def get_metadata(self, dataset_id: str, indices: List[int]) -> List[dict]:
        """Get metadata for specific indices

        Args:
            dataset_id: Dataset ID
            indices: List of indices

        Returns:
            List of metadata dicts
        """
        # Load dataset if needed
        _, metadata = self.load_dataset(dataset_id)

        # Return metadata for indices
        results = []
        for idx in indices:
            if 0 <= idx < len(metadata):
                results.append(metadata[idx])
            else:
                results.append({"error": "Index out of bounds"})

        return results

    def unload_dataset(self, dataset_id: str):
        """Unload a dataset from memory"""
        if dataset_id in self.indices:
            del self.indices[dataset_id]
            del self.metadata[dataset_id]
            print(f"✓ Dataset {dataset_id} unloaded")

    def get_stats(self) -> dict:
        """Get statistics about loaded datasets"""
        return {
            "total_datasets": len(self.datasets_info),
            "loaded_datasets": len(self.indices),
            "datasets": self.datasets_info
        }


# Global instance
embedding_manager = EmbeddingManager()
