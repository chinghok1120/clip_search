#!/usr/bin/env python3
"""
Benchmark image encoding performance
Tests different batch sizes, image counts, and resolutions

Usage:
    python benchmark_encoding.py --num-images 100 --resolutions 640x360,1920x1080
"""

import argparse
import time
import tempfile
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm


class EncodingBenchmark:
    """Benchmark image encoding performance"""

    def __init__(self, model_name='EVA02-B-16', pretrained='merged2b_s8b_b131k', device='cuda'):
        print(f"Loading model: {model_name}")
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=device
        )
        self.model.eval()

        # Get embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224).to(device)
            self.embedding_dim = self.model.encode_image(dummy).shape[-1]

        print(f"✓ Model loaded")
        print(f"  Device: {device}")
        print(f"  Embedding dimension: {self.embedding_dim}")

    def create_test_images(self, num_images: int, resolution: Tuple[int, int]) -> List[Image.Image]:
        """Create test images with random content"""
        print(f"\nCreating {num_images} test images ({resolution[0]}×{resolution[1]})...")
        images = []
        for i in range(num_images):
            # Create random image
            img_array = np.random.randint(0, 255, (*resolution[::-1], 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            images.append(img)
        return images

    def benchmark_batch_sizes(self, images: List[Image.Image], batch_sizes: List[int]):
        """Benchmark different batch sizes"""
        print(f"\n{'='*80}")
        print(f"Benchmarking Batch Sizes ({len(images)} images)")
        print(f"{'='*80}")

        results = []

        for batch_size in batch_sizes:
            # Process in batches
            embeddings_list = []
            times = []

            num_batches = (len(images) + batch_size - 1) // batch_size

            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]

                # Preprocess
                batch_tensor = torch.stack([self.preprocess(img) for img in batch_images])
                batch_tensor = batch_tensor.to(self.device)

                # Warmup on first batch
                if i == 0:
                    with torch.no_grad():
                        _ = self.model.encode_image(batch_tensor)
                    if self.device == 'cuda':
                        torch.cuda.synchronize()

                # Encode and time
                start = time.time()
                with torch.no_grad():
                    features = self.model.encode_image(batch_tensor)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.time() - start
                times.append(elapsed)

                embeddings_list.append(features.cpu().numpy())

            # Calculate stats
            total_time = sum(times)
            avg_time_per_batch = np.mean(times) * 1000  # ms
            time_per_image = (total_time / len(images)) * 1000  # ms
            throughput = len(images) / total_time  # img/sec
            throughput_min = throughput * 60  # img/min

            results.append({
                'batch_size': batch_size,
                'num_images': len(images),
                'total_time': total_time,
                'time_per_image': time_per_image,
                'throughput_sec': throughput,
                'throughput_min': throughput_min
            })

            print(f"\nBatch size: {batch_size}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Time per image: {time_per_image:.2f}ms")
            print(f"  Throughput: {throughput:.1f} img/sec = {throughput_min:.0f} img/min")

        return results

    def benchmark_resolutions(self, num_images: int, resolutions: List[Tuple[int, int]], batch_size: int = 16):
        """Benchmark different image resolutions"""
        print(f"\n{'='*80}")
        print(f"Benchmarking Image Resolutions ({num_images} images, batch_size={batch_size})")
        print(f"{'='*80}")

        results = []

        for resolution in resolutions:
            # Create test images
            images = self.create_test_images(num_images, resolution)

            # Encode
            start = time.time()
            embeddings_list = []

            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                batch_tensor = torch.stack([self.preprocess(img) for img in batch_images])
                batch_tensor = batch_tensor.to(self.device)

                with torch.no_grad():
                    features = self.model.encode_image(batch_tensor)
                    embeddings_list.append(features.cpu().numpy())

            if self.device == 'cuda':
                torch.cuda.synchronize()

            total_time = time.time() - start
            time_per_image = (total_time / num_images) * 1000
            throughput = num_images / total_time
            throughput_min = throughput * 60

            results.append({
                'resolution': f"{resolution[0]}×{resolution[1]}",
                'num_images': num_images,
                'total_time': total_time,
                'time_per_image': time_per_image,
                'throughput_min': throughput_min
            })

            print(f"\nResolution: {resolution[0]}×{resolution[1]}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Time per image: {time_per_image:.2f}ms")
            print(f"  Throughput: {throughput:.1f} img/sec = {throughput_min:.0f} img/min")

        return results

    def print_summary(self, batch_results, resolution_results):
        """Print summary table"""
        print(f"\n{'='*80}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*80}")

        print("\n1. Batch Size Impact:")
        print(f"{'Batch Size':>12} | {'Images/Min':>12} | {'ms/Image':>10} | {'Speedup':>8}")
        print("-" * 60)
        baseline = batch_results[0]['throughput_min']
        for r in batch_results:
            speedup = r['throughput_min'] / baseline
            print(f"{r['batch_size']:>12} | {r['throughput_min']:>12.0f} | {r['time_per_image']:>10.2f} | {speedup:>8.2f}x")

        print("\n2. Resolution Impact:")
        print(f"{'Resolution':>12} | {'Images/Min':>12} | {'ms/Image':>10}")
        print("-" * 60)
        for r in resolution_results:
            print(f"{r['resolution']:>12} | {r['throughput_min']:>12.0f} | {r['time_per_image']:>10.2f}")

        print("\n3. 32-Camera Requirement Check:")
        target = 960  # images/min for 32 cameras
        best = max(batch_results, key=lambda x: x['throughput_min'])
        print(f"  Target: {target} img/min (32 cameras @ 2-sec sampling)")
        print(f"  Best achieved: {best['throughput_min']:.0f} img/min (batch_size={best['batch_size']})")
        if best['throughput_min'] >= target:
            margin = (best['throughput_min'] / target - 1) * 100
            print(f"  ✓ PASS: {margin:.1f}% above target")
        else:
            shortfall = (1 - best['throughput_min'] / target) * 100
            print(f"  ✗ FAIL: {shortfall:.1f}% below target")

        print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark image encoding')
    parser.add_argument('--num-images', type=int, default=100,
                        help='Number of test images (default: 100)')
    parser.add_argument('--batch-sizes', type=str, default='1,4,8,16,32',
                        help='Batch sizes to test (default: 1,4,8,16,32)')
    parser.add_argument('--resolutions', type=str, default='640x360,1920x1080,3840x2160',
                        help='Resolutions to test (default: 640x360,1920x1080,3840x2160)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device (default: cuda)')
    parser.add_argument('--model', type=str, default='EVA02-B-16',
                        help='Model name (default: EVA02-B-16)')

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    # Parse resolutions
    resolutions = []
    for res in args.resolutions.split(','):
        w, h = res.split('x')
        resolutions.append((int(w), int(h)))

    print(f"{'='*80}")
    print(f"IMAGE ENCODING BENCHMARK")
    print(f"{'='*80}")
    print(f"Test images: {args.num_images}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Resolutions: {[f'{w}×{h}' for w, h in resolutions]}")
    print(f"Device: {args.device}")

    # Initialize benchmark
    benchmark = EncodingBenchmark(model_name=args.model, device=args.device)

    # Create test images (use first resolution for batch size test)
    test_images = benchmark.create_test_images(args.num_images, resolutions[0])

    # Benchmark batch sizes
    batch_results = benchmark.benchmark_batch_sizes(test_images, batch_sizes)

    # Benchmark resolutions
    resolution_results = benchmark.benchmark_resolutions(args.num_images, resolutions, batch_size=16)

    # Print summary
    benchmark.print_summary(batch_results, resolution_results)


if __name__ == '__main__':
    main()
