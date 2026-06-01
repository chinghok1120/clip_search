#!/usr/bin/env python3
"""
Encode images from a folder using EVA-02-B/14 model
Stores embeddings in FAISS index for fast similarity search

Usage:
    python encode_images.py --input /path/to/images --output /path/to/output

Output:
    - embeddings.index (FAISS index)
    - metadata.json (image paths and info)
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm

# Try to import FAISS (will guide installation if not available)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️  FAISS not installed. Install with: pip install faiss-gpu")
    FAISS_AVAILABLE = False


class ImageEncoder:
    """Encode images using CLIP model"""

    def __init__(self, model_name='EVA02-B-16', pretrained='merged2b_s8b_b131k', device='cuda', precision='fp32'):
        print(f"Loading model: {model_name}")
        print(f"Device: {device}")

        self.device = device
        self.precision = precision
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device
        )
        self.model.eval()

        # Get embedding dimension using the model's own reported image size
        image_size = getattr(self.model.visual, 'image_size', 224)
        if isinstance(image_size, (tuple, list)):
            image_size = image_size[0]
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size).to(device)
            self.embedding_dim = self.model.encode_image(dummy).shape[-1]

        print(f"✓ Model loaded, embedding dimension: {self.embedding_dim}")

    def load_image(self, image_path: Path) -> Image.Image:
        """Load and validate image"""
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            raise ValueError(f"Failed to load {image_path}: {e}")

    def encode_batch(self, image_paths: List[Path], batch_size: int = 16) -> Tuple[np.ndarray, List[Dict]]:
        """
        Encode a batch of images

        Args:
            image_paths: List of image paths
            batch_size: Number of images to process at once

        Returns:
            embeddings: numpy array of shape (N, embedding_dim)
            metadata: list of dicts with image info
        """
        embeddings_list = []
        metadata_list = []
        failed_images = []

        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            batch_metadata = []

            # Load images
            for img_path in batch_paths:
                try:
                    img = self.load_image(img_path)
                    batch_images.append(self.preprocess(img))
                    batch_metadata.append({
                        'path': str(img_path),
                        'filename': img_path.name,
                        'size': img.size,
                    })
                except Exception as e:
                    failed_images.append((str(img_path), str(e)))
                    continue

            if not batch_images:
                continue

            # Encode batch
            batch_tensor = torch.stack(batch_images).to(self.device)

            with torch.no_grad():
                if self.precision == 'fp16' and self.device == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        features = self.model.encode_image(batch_tensor)
                    features = features.float()
                else:
                    features = self.model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)  # Normalize
                embeddings_list.append(features.cpu().numpy())

            metadata_list.extend(batch_metadata)

        if failed_images:
            print(f"\n⚠️  Failed to process {len(failed_images)} images:")
            for path, error in failed_images[:5]:  # Show first 5
                print(f"  - {path}: {error}")
            if len(failed_images) > 5:
                print(f"  ... and {len(failed_images) - 5} more")

        # Concatenate all embeddings
        if embeddings_list:
            embeddings = np.vstack(embeddings_list)
        else:
            embeddings = np.array([])

        return embeddings, metadata_list


def model_file_suffix(model_name: str) -> str:
    """Convert OpenCLIP model name to file suffix: EVA02-B-16 → eva02-b-16"""
    return model_name.lower().replace("_", "-")


def save_embeddings_faiss(embeddings: np.ndarray, metadata: List[Dict], output_base: Path, model_name: str):
    """
    Save embeddings as {output_base}_{model_suffix}.faiss and .json

    Args:
        embeddings: numpy array of shape (N, dim)
        metadata: list of metadata dicts
        output_base: base path, e.g. /path/to/embeddings/crowdhuman
        model_name: OpenCLIP model name, e.g. EVA02-B-16
    """
    if not FAISS_AVAILABLE:
        print("❌ FAISS not available. Install with: pip install faiss-gpu")
        print("   Falling back to NumPy storage...")
        save_embeddings_numpy(embeddings, metadata, output_base, model_name)
        return

    suffix = model_file_suffix(model_name)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    # Create FAISS index
    print(f"\nCreating FAISS index...")
    embedding_dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings.astype('float32'))
    print(f"✓ Added {index.ntotal} embeddings to FAISS index")

    # Save as {dataset}_{model_suffix}.faiss  e.g. crowdhuman_eva02-b-16.faiss
    index_path = output_base.parent / f"{output_base.name}_{suffix}.faiss"
    faiss.write_index(index, str(index_path))
    print(f"✓ Saved FAISS index: {index_path}")

    # Save metadata as {dataset}_{model_suffix}.json
    metadata_path = output_base.parent / f"{output_base.name}_{suffix}.json"
    with open(metadata_path, 'w') as f:
        json.dump({
            'num_embeddings': len(embeddings),
            'embedding_dim': embedding_dim,
            'model': model_name,
            'images': metadata
        }, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")

    return index_path, metadata_path


def save_embeddings_numpy(embeddings: np.ndarray, metadata: List[Dict], output_base: Path, model_name: str):
    """Fallback: save as {dataset}_{model_suffix}.npz"""
    suffix = model_file_suffix(model_name)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_base.parent / f"{output_base.name}_{suffix}.npz"
    np.savez_compressed(output_path, embeddings=embeddings, metadata=np.array(metadata, dtype=object))
    return output_path, output_path


def find_images(input_dir: Path) -> List[Path]:
    """Find all image files in directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(input_dir.glob(f'**/*{ext}'))
        image_paths.extend(input_dir.glob(f'**/*{ext.upper()}'))

    return sorted(image_paths)


def main():
    parser = argparse.ArgumentParser(description='Encode images using CLIP model')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing images')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for embeddings')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for encoding (default: 16)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--model', type=str, default='EVA02-B-16',
                        help='OpenCLIP model name (default: EVA02-B-16)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights name. Defaults: EVA02-B-16=merged2b_s8b_b131k, EVA02-L-14-336=merged2b_s6b_b61k')
    parser.add_argument('--format', type=str, default='faiss',
                        choices=['faiss', 'numpy'],
                        help='Output format (default: faiss)')
    parser.add_argument('--precision', type=str, default='fp32',
                        choices=['fp32', 'fp16'],
                        help='Inference precision; fp16 roughly halves VRAM (recommended for ViT-bigG)')

    args = parser.parse_args()

    # Validate paths
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        return

    # Resolve default pretrained weights per model
    pretrained_defaults = {
        'EVA02-B-16': 'merged2b_s8b_b131k',
        'EVA02-L-14-336': 'merged2b_s6b_b61k',
    }
    pretrained = args.pretrained or pretrained_defaults.get(args.model, 'openai')
    print(f"Pretrained weights: {pretrained}")

    # --output is now a base path: directory + dataset name (no extension)
    # Files will be saved as {output}_{model_suffix}.faiss / .json
    # e.g. --output ~/embeddings/crowdhuman  →  crowdhuman_eva02-b-16.faiss
    output_base = Path(args.output)

    # Find images
    print(f"Scanning for images in: {input_dir}")
    image_paths = find_images(input_dir)
    print(f"✓ Found {len(image_paths)} images")

    if not image_paths:
        print("❌ No images found!")
        return

    # Initialize encoder
    encoder = ImageEncoder(model_name=args.model, pretrained=pretrained, device=args.device, precision=args.precision)

    # Encode images
    print(f"\n{'='*80}")
    print(f"ENCODING STARTED")
    print(f"{'='*80}")

    encoding_start = time.time()
    embeddings, metadata = encoder.encode_batch(image_paths, batch_size=args.batch_size)
    encoding_end = time.time()
    encoding_time = encoding_end - encoding_start

    # Save embeddings
    save_start = time.time()
    if args.format == 'faiss':
        index_path, metadata_path = save_embeddings_faiss(embeddings, metadata, output_base, args.model)
    else:
        output_path, _ = save_embeddings_numpy(embeddings, metadata, output_base, args.model)
        index_path = output_path
        metadata_path = output_path
    save_end = time.time()
    save_time = save_end - save_start

    total_time = save_end - encoding_start
    failed_count = len(image_paths) - len(embeddings)

    # Print comprehensive statistics
    print(f"\n{'='*80}")
    print(f"ENCODING STATISTICS")
    print(f"{'='*80}")
    print(f"\n📊 Input:")
    print(f"  Images found:        {len(image_paths)}")
    print(f"  Images processed:    {len(embeddings)}")
    print(f"  Failed:              {failed_count}")
    print(f"  Success rate:        {len(embeddings)/len(image_paths)*100:.1f}%")

    print(f"\n⚙️  Configuration:")
    print(f"  Model:               {args.model}")
    print(f"  Device:              {args.device}")
    print(f"  Precision:           {args.precision}")
    print(f"  Batch size:          {args.batch_size}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")

    print(f"\n⏱️  Performance:")
    print(f"  Encoding time:       {encoding_time:.2f}s")
    print(f"  Saving time:         {save_time:.2f}s")
    print(f"  Total time:          {total_time:.2f}s")
    print(f"  Time per image:      {encoding_time/len(embeddings)*1000:.2f}ms")
    print(f"  Throughput:          {len(embeddings)/encoding_time:.1f} img/sec")
    print(f"  Throughput:          {len(embeddings)/encoding_time*60:.0f} img/min")

    print(f"\n💾 Output:")
    if args.format == 'faiss':
        index_size = index_path.stat().st_size
        metadata_size = metadata_path.stat().st_size
        total_size = index_size + metadata_size
        print(f"  Index file:          {index_path}")
        print(f"  Index size:          {index_size / 1024 / 1024:.2f} MB")
        print(f"  Metadata file:       {metadata_path}")
        print(f"  Metadata size:       {metadata_size / 1024:.2f} KB")
        print(f"  Total size:          {total_size / 1024 / 1024:.2f} MB")
        print(f"  Bytes per embedding: {total_size / len(embeddings):.0f} bytes")
    else:
        file_size = index_path.stat().st_size
        print(f"  Output file:         {index_path}")
        print(f"  File size:           {file_size / 1024 / 1024:.2f} MB")
        print(f"  Bytes per embedding: {file_size / len(embeddings):.0f} bytes")

    print(f"\n✅ ENCODING COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
