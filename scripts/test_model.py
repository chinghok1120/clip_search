#!/usr/bin/env python3
"""
Test script for EVA-02-B/14 model
Tests model loading, CUDA availability, and basic inference
"""

import time
import torch
import open_clip
from PIL import Image
import numpy as np

def print_separator(title=""):
    print("\n" + "="*60)
    if title:
        print(f"  {title}")
        print("="*60)

def check_cuda():
    """Check CUDA availability and GPU info"""
    print_separator("CUDA & GPU Information")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️  CUDA not available - will run on CPU (slower)")
        print("   To use GPU: ensure NVIDIA driver is loaded (sudo modprobe nvidia)")

    return torch.cuda.is_available()

def load_model(model_name='EVA02-B-16', pretrained='merged2b_s8b_b131k'):
    """Load EVA-02-B/14 model from OpenCLIP"""
    print_separator(f"Loading Model: {model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Model: {model_name}")
    print(f"Pretrained weights: {pretrained}")
    print("Loading... (this may take a minute on first run)")

    start = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    load_time = time.time() - start

    print(f"✓ Model loaded in {load_time:.2f} seconds")

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    return model, preprocess, tokenizer, device

def create_test_image(size=(224, 224)):
    """Create a test image (random noise)"""
    # Create a simple test image with some patterns
    img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    # Add some structure (diagonal stripes)
    for i in range(size[0]):
        for j in range(size[1]):
            if (i + j) % 20 < 10:
                img_array[i, j] = [255, 100, 100]  # Red-ish

    return Image.fromarray(img_array)

def benchmark_image_encoding(model, preprocess, device, num_runs=10):
    """Benchmark image encoding speed"""
    print_separator("Image Encoding Benchmark")

    # Create test image
    test_image = create_test_image()
    processed_image = preprocess(test_image).unsqueeze(0).to(device)

    print(f"Input shape: {processed_image.shape}")
    print(f"Running {num_runs} iterations...")

    # Warmup
    with torch.no_grad():
        _ = model.encode_image(processed_image)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for i in range(num_runs):
            start = time.time()
            features = model.encode_image(processed_image)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed * 1000)  # Convert to ms

            if i == 0:
                print(f"Output shape: {features.shape}")

    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"\nResults (single image):")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Std Dev: {std_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    print(f"  Throughput: {1000/avg_time:.1f} images/sec")

    return avg_time, features.shape[-1]

def benchmark_batch_encoding(model, preprocess, device, batch_sizes=[1, 4, 8, 16]):
    """Benchmark batch image encoding"""
    print_separator("Batch Encoding Benchmark")

    results = {}

    for batch_size in batch_sizes:
        # Create batch of test images
        test_image = create_test_image()
        batch = torch.stack([preprocess(test_image) for _ in range(batch_size)]).to(device)

        # Warmup
        with torch.no_grad():
            _ = model.encode_image(batch)

        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark (3 runs)
        times = []
        with torch.no_grad():
            for _ in range(3):
                start = time.time()
                _ = model.encode_image(batch)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - start)

        avg_time = np.mean(times) * 1000  # ms
        time_per_image = avg_time / batch_size
        throughput = 1000 / time_per_image

        results[batch_size] = {
            'total_time': avg_time,
            'time_per_image': time_per_image,
            'throughput': throughput
        }

        print(f"Batch size {batch_size:2d}: {avg_time:6.1f}ms total | "
              f"{time_per_image:5.1f}ms/img | {throughput:6.1f} img/sec | "
              f"{throughput*60:6.0f} img/min")

    return results

def benchmark_text_encoding(model, tokenizer, device, num_runs=10):
    """Benchmark text encoding speed"""
    print_separator("Text Encoding Benchmark")

    test_queries = [
        "person in red jacket",
        "woman in blue dress",
        "car in parking lot",
        "man walking with dog",
        "delivery truck at loading dock"
    ]

    print(f"Test queries: {len(test_queries)}")
    print(f"Running {num_runs} iterations...")

    # Tokenize
    text_tokens = tokenizer(test_queries).to(device)
    print(f"Token shape: {text_tokens.shape}")

    # Warmup
    with torch.no_grad():
        _ = model.encode_text(text_tokens)

    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for i in range(num_runs):
            start = time.time()
            text_features = model.encode_text(text_tokens)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed * 1000)  # ms

            if i == 0:
                print(f"Output shape: {text_features.shape}")

    avg_time = np.mean(times)
    avg_per_query = avg_time / len(test_queries)

    print(f"\nResults ({len(test_queries)} queries):")
    print(f"  Average total: {avg_time:.2f} ms")
    print(f"  Average per query: {avg_per_query:.2f} ms")

    return avg_time, avg_per_query

def test_similarity_search(model, preprocess, tokenizer, device):
    """Test image-text similarity (basic search functionality)"""
    print_separator("Image-Text Similarity Test")

    # Create test image
    test_image = create_test_image()
    image_input = preprocess(test_image).unsqueeze(0).to(device)

    # Test queries
    queries = [
        "red and pink pattern",
        "blue ocean waves",
        "green forest",
        "abstract geometric shapes",
        "diagonal stripes"
    ]

    text_tokens = tokenizer(queries).to(device)

    # Encode
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (image_features @ text_features.T).squeeze(0)
        similarity = similarity.cpu().numpy()

    print("Query similarities (higher = better match):")
    for i, query in enumerate(queries):
        print(f"  '{query}': {similarity[i]:.3f}")

    best_match_idx = np.argmax(similarity)
    print(f"\n✓ Best match: '{queries[best_match_idx]}' (score: {similarity[best_match_idx]:.3f})")

def main():
    print("\n" + "█"*60)
    print("  EVA-02-B/14 Model Test Script")
    print("  CLIP-based Surveillance Search System")
    print("█"*60)

    # Check CUDA
    cuda_available = check_cuda()

    # Load model
    try:
        model, preprocess, tokenizer, device = load_model(
            model_name='EVA02-B-16',
            pretrained='merged2b_s8b_b131k'
        )
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nTrying alternative model name...")
        try:
            model, preprocess, tokenizer, device = load_model(
                model_name='EVA02-B-16',
                pretrained='laion2b_s9b_b144k'
            )
        except Exception as e2:
            print(f"❌ Failed again: {e2}")
            return

    # Benchmark image encoding
    avg_time, embedding_dim = benchmark_image_encoding(model, preprocess, device, num_runs=10)

    # Benchmark batch encoding
    batch_results = benchmark_batch_encoding(model, preprocess, device, batch_sizes=[1, 4, 8, 16])

    # Benchmark text encoding
    text_time, per_query_time = benchmark_text_encoding(model, tokenizer, device, num_runs=10)

    # Test similarity
    test_similarity_search(model, preprocess, tokenizer, device)

    # Summary
    print_separator("Summary")
    print(f"Model: EVA02-B-16")
    print(f"Device: {device.upper()}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"\nPerformance:")
    print(f"  Single image: {avg_time:.1f} ms ({1000/avg_time:.1f} img/sec)")
    print(f"  Batch of 16:  {batch_results[16]['time_per_image']:.1f} ms/img ({batch_results[16]['throughput']*60:.0f} img/min)")
    print(f"  Text query:   {per_query_time:.1f} ms")

    # Check requirements
    print(f"\n32-camera requirement check:")
    required_throughput = 960  # images per minute
    actual_throughput = batch_results[16]['throughput'] * 60

    if actual_throughput >= required_throughput:
        margin = (actual_throughput / required_throughput - 1) * 100
        print(f"  ✓ PASS: {actual_throughput:.0f} img/min (target: {required_throughput} img/min)")
        print(f"  Safety margin: {margin:.1f}%")
    else:
        shortfall = (1 - actual_throughput / required_throughput) * 100
        print(f"  ✗ FAIL: {actual_throughput:.0f} img/min (target: {required_throughput} img/min)")
        print(f"  Shortfall: {shortfall:.1f}%")
        max_cameras = int(32 * actual_throughput / required_throughput)
        print(f"  Max cameras at 2-sec sampling: ~{max_cameras}")

    print("\n" + "█"*60)
    print("  Test Complete!")
    print("█"*60 + "\n")

if __name__ == "__main__":
    main()
