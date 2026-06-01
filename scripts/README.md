# Scripts Directory

Test and utility scripts for the CLIP-based surveillance search system.

## Available Scripts

### `test_model.py`

Comprehensive test script for EVA-02-B/14 model.

**Features**:
- CUDA/GPU availability check
- Model loading from OpenCLIP
- Single image encoding benchmark
- Batch encoding benchmark (1, 4, 8, 16 images)
- Text query encoding benchmark
- Image-text similarity test
- Requirements validation (960 img/min target)

**Usage**:
```bash
source ../venv/bin/activate
python test_model.py
```

**Expected Output**:
- Model info (parameters, embedding dimension)
- Performance metrics (ms per image, throughput)
- Pass/fail for 32-camera requirement

**Requirements**:
- `torch`, `open-clip-torch`, `pillow`, `numpy`
- GPU optional but recommended (5-10× faster)

## Setup

1. **Create virtual environment** (from project root):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install open-clip-torch pillow numpy
   ```

3. **Run tests**:
   ```bash
   cd scripts
   python test_model.py
   ```

## Benchmark Results

### Linux PC (CPU - 12 cores)
- Single image: ~62ms
- Batch of 16: ~57ms/image (1,058 img/min) ✓
- Status: **Meets requirements**

### Linux PC (GPU - RTX 3090) - Expected
- Single image: ~10ms (estimated)
- Batch of 16: ~20ms/image (3,000 img/min) ✓
- Status: **Should exceed requirements**

### Jetson Orin Nano - Target Platform
- Single image: ~70ms (estimated)
- Batch of 16: ~60ms/image (1,000 img/min) ✓
- Status: **Should meet requirements**

## Next Steps

1. **Enable GPU**: Load NVIDIA driver (`sudo modprobe nvidia`)
2. **Test on real thumbnails**: Use actual surveillance camera snapshots
3. **Test different batch sizes**: Find optimal batch size for throughput
4. **Profile memory usage**: Ensure fits within budget
5. **Export to ONNX/TensorRT**: Optimize for production deployment
