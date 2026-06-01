#!/usr/bin/env python3
"""
Search images using text queries on pre-computed embeddings

Usage:
    python search_images.py --embeddings /path/to/embeddings --query "woman in red dress" --top-k 10

Requires:
    - embeddings.index (FAISS index) or embeddings.npz (NumPy)
    - metadata.json
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import open_clip

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class ImageSearcher:
    """Search pre-computed image embeddings using text queries"""

    def __init__(self, embeddings_dir: Path, model_name='EVA02-B-16',
                 pretrained='merged2b_s8b_b131k', device='cuda'):
        """
        Initialize searcher

        Args:
            embeddings_dir: Directory containing embeddings and metadata
            model_name: OpenCLIP model name (must match encoding model)
            pretrained: Pretrained weights
            device: 'cuda' or 'cpu'
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.device = device

        # Load model for text encoding
        print(f"Loading model: {model_name}")
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        print("✓ Model loaded")

        # Load embeddings and metadata
        self.load_embeddings()

    def load_embeddings(self):
        """Load pre-computed embeddings"""
        # Try FAISS first
        faiss_path = self.embeddings_dir / 'embeddings.index'
        numpy_path = self.embeddings_dir / 'embeddings.npz'
        metadata_path = self.embeddings_dir / 'metadata.json'

        if faiss_path.exists() and FAISS_AVAILABLE:
            print(f"Loading FAISS index: {faiss_path}")
            self.index = faiss.read_index(str(faiss_path))
            self.use_faiss = True
            print(f"✓ Loaded {self.index.ntotal} embeddings")

        elif numpy_path.exists():
            print(f"Loading NumPy embeddings: {numpy_path}")
            data = np.load(numpy_path, allow_pickle=True)
            self.embeddings = data['embeddings']
            self.use_faiss = False
            print(f"✓ Loaded {len(self.embeddings)} embeddings")

        else:
            raise FileNotFoundError(
                f"No embeddings found in {self.embeddings_dir}\n"
                f"Expected: {faiss_path} or {numpy_path}"
            )

        # Load metadata
        if metadata_path.exists():
            print(f"Loading metadata: {metadata_path}")
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.images = self.metadata['images']
            print(f"✓ Loaded metadata for {len(self.images)} images")
        else:
            # Fallback if using numpy without metadata.json
            if not self.use_faiss:
                self.images = data['metadata'].tolist()
                print(f"✓ Loaded metadata from NumPy file")
            else:
                raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    def encode_text(self, query: str) -> np.ndarray:
        """Encode text query to embedding"""
        with torch.no_grad():
            text_tokens = self.tokenizer([query]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.cpu().numpy()

    def search_faiss(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS index"""
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        return scores[0], indices[0]

    def search_numpy(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Search using NumPy (linear search)"""
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_embedding.T).squeeze()

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_scores = similarities[top_indices]

        return top_scores, top_indices

    def search(self, query: str, top_k: int = 10) -> Tuple[List[dict], dict]:
        """
        Search for images matching text query

        Args:
            query: Text query (e.g., "woman in red dress")
            top_k: Number of results to return

        Returns:
            Tuple of (results, timing_stats)
        """
        # Encode query
        text_encode_start = time.time()
        query_embedding = self.encode_text(query)
        text_encode_time = time.time() - text_encode_start

        # Search
        search_start = time.time()
        if self.use_faiss:
            scores, indices = self.search_faiss(query_embedding, top_k)
        else:
            scores, indices = self.search_numpy(query_embedding, top_k)
        search_time = time.time() - search_start

        # Format results
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), 1):
            result = {
                'rank': rank,
                'score': float(score),
                'image': self.images[int(idx)]
            }
            results.append(result)

        # Collect timing stats
        timing_stats = {
            'text_encode_time': text_encode_time,
            'search_time': search_time,
            'total_time': text_encode_time + search_time
        }

        return results, timing_stats

    def print_results(self, results: List[dict], show_path: bool = True):
        """Pretty print search results"""
        print(f"\n{'='*80}")
        print(f"  Search Results ({len(results)} matches)")
        print(f"{'='*80}")

        for result in results:
            score_pct = result['score'] * 100
            print(f"\n#{result['rank']}  Score: {score_pct:.1f}%")

            if show_path:
                print(f"  Path: {result['image']['path']}")
            else:
                print(f"  File: {result['image']['filename']}")

            if 'size' in result['image']:
                print(f"  Size: {result['image']['size']}")

        print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Search images using text queries')
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Directory containing embeddings and metadata')
    parser.add_argument('--query', type=str, required=True,
                        help='Text query (e.g., "woman in red dress")')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of results to return (default: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--model', type=str, default='EVA02-B-16',
                        help='OpenCLIP model name (must match encoding)')
    parser.add_argument('--show-full-path', action='store_true',
                        help='Show full paths instead of filenames')
    parser.add_argument('--output', type=str, default=None,
                        help='Output folder to copy matched images (optional)')
    parser.add_argument('--json', action='store_true',
                        help='Output results as JSON')

    args = parser.parse_args()

    # Validate embeddings directory
    embeddings_dir = Path(args.embeddings)
    if not embeddings_dir.exists():
        print(f"❌ Embeddings directory does not exist: {embeddings_dir}")
        return

    # Get file sizes
    faiss_path = embeddings_dir / 'embeddings.index'
    numpy_path = embeddings_dir / 'embeddings.npz'
    metadata_path = embeddings_dir / 'metadata.json'

    if faiss_path.exists():
        index_size = faiss_path.stat().st_size
        metadata_size = metadata_path.stat().st_size if metadata_path.exists() else 0
        total_db_size = index_size + metadata_size
        db_type = 'FAISS'
    elif numpy_path.exists():
        index_size = numpy_path.stat().st_size
        metadata_size = 0
        total_db_size = index_size
        db_type = 'NumPy'
    else:
        index_size = 0
        metadata_size = 0
        total_db_size = 0
        db_type = 'Unknown'

    # Initialize searcher
    print(f"\n{'='*80}")
    print(f"INITIALIZING SEARCH")
    print(f"{'='*80}")

    searcher = ImageSearcher(
        embeddings_dir=embeddings_dir,
        model_name=args.model,
        device=args.device
    )

    # Get embedding dimension
    if searcher.use_faiss:
        embedding_dim = searcher.index.d
        num_embeddings = searcher.index.ntotal
    else:
        embedding_dim = searcher.embeddings.shape[1]
        num_embeddings = len(searcher.embeddings)

    # Search
    print(f"\n{'='*80}")
    print(f"SEARCHING")
    print(f"{'='*80}")
    print(f"Query: '{args.query}'")
    print(f"Top-K: {args.top_k}")

    results, timing_stats = searcher.search(args.query, top_k=args.top_k)

    # Output results
    if args.json:
        import json
        output = {
            'query': args.query,
            'results': results,
            'stats': {
                'num_results': len(results),
                'text_encode_time_ms': timing_stats['text_encode_time'] * 1000,
                'search_time_ms': timing_stats['search_time'] * 1000,
                'total_time_ms': timing_stats['total_time'] * 1000,
                'database_type': db_type,
                'total_embeddings': num_embeddings,
                'embedding_dim': embedding_dim
            }
        }
        print(json.dumps(output, indent=2))
    else:
        # Print comprehensive statistics
        print(f"\n{'='*80}")
        print(f"SEARCH STATISTICS")
        print(f"{'='*80}")

        print(f"\n📝 Query:")
        print(f"  Text:                '{args.query}'")
        print(f"  Top-K requested:     {args.top_k}")
        print(f"  Results returned:    {len(results)}")

        print(f"\n💾 Database:")
        print(f"  Type:                {db_type}")
        print(f"  Total embeddings:    {num_embeddings:,}")
        print(f"  Embedding dimension: {embedding_dim}")
        if db_type == 'FAISS':
            print(f"  Index file:          {faiss_path}")
            print(f"  Index size:          {index_size / 1024 / 1024:.2f} MB")
            if metadata_size > 0:
                print(f"  Metadata file:       {metadata_path}")
                print(f"  Metadata size:       {metadata_size / 1024:.2f} KB")
        else:
            print(f"  Data file:           {numpy_path}")
            print(f"  File size:           {index_size / 1024 / 1024:.2f} MB")
        print(f"  Total size:          {total_db_size / 1024 / 1024:.2f} MB")
        print(f"  Bytes per embedding: {total_db_size / num_embeddings:.0f} bytes")

        print(f"\n⏱️  Performance:")
        print(f"  Text encoding:       {timing_stats['text_encode_time']*1000:.2f}ms")
        print(f"  Vector search:       {timing_stats['search_time']*1000:.2f}ms")
        print(f"  Total time:          {timing_stats['total_time']*1000:.2f}ms")

        if results:
            scores = [r['score'] for r in results]
            print(f"\n📊 Results:")
            print(f"  Top score:           {max(scores)*100:.1f}%")
            print(f"  Lowest score:        {min(scores)*100:.1f}%")
            print(f"  Average score:       {sum(scores)/len(scores)*100:.1f}%")

        print(f"\n⚙️  Configuration:")
        print(f"  Model:               {args.model}")
        print(f"  Device:              {args.device}")

        print(f"\n{'='*80}")

        # Copy images to output folder if specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n📁 Copying matched images to: {output_dir}")
            copied_count = 0

            for result in results:
                src_path = Path(result['image']['path'])
                if src_path.exists():
                    # Create filename with rank prefix: 01_original.jpg, 02_original.jpg, etc.
                    rank_prefix = f"{result['rank']:02d}"
                    score_str = f"{result['score']*100:.1f}"
                    dst_filename = f"{rank_prefix}_score{score_str}_{src_path.name}"
                    dst_path = output_dir / dst_filename

                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                    print(f"  ✓ Copied: {dst_filename}")
                else:
                    print(f"  ✗ Not found: {src_path}")

            print(f"\n✅ Copied {copied_count}/{len(results)} images to {output_dir}")

        # Print results
        searcher.print_results(results, show_path=args.show_full_path)


if __name__ == '__main__':
    main()
