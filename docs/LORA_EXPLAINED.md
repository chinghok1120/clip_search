# LoRA (Low-Rank Adaptation) Explained

Understanding how LoRA lets you add new knowledge without changing the base model.

---

## The Core Idea

**Problem:** Fine-tuning changes all model weights → forgets old knowledge

**LoRA Solution:** Keep base model frozen, add tiny "adapter" layers that learn new knowledge

**Analogy:**
```
Traditional fine-tuning:
  Rewrite the entire textbook ❌
  (might lose original content)

LoRA:
  Keep textbook unchanged ✓
  Add sticky notes with new info ✓
  (original + new knowledge preserved)
```

---

## How LoRA Works (Simple Explanation)

### Traditional Fine-Tuning

**Base model has weight matrices:**
```
W = [512 x 512] matrix  (262,144 parameters)

During fine-tuning:
  W_new = W + ΔW
  
All 262,144 parameters change!
```

**Problem:**
- Changing all weights is expensive (memory, compute)
- Easy to overwrite/forget original knowledge
- Need to save entire model again (365MB)

---

### LoRA Approach

**Instead of changing W, add a small adapter:**

```
Original:
  Output = W × Input
  
LoRA:
  Output = W × Input  +  (A × B) × Input
           ↑               ↑
        frozen          trainable
      (365MB)            (10MB)
```

**Key insight: A and B are LOW-RANK matrices**

```
W = [512 x 512]  →  262,144 parameters (frozen)

LoRA decomposes update into:
  A = [512 x 16]  →  8,192 parameters
  B = [16 x 512]  →  8,192 parameters
  
Total trainable: 16,384 parameters (16× smaller!)
```

**Why this works:**
- Most weight updates are "low-rank" (can be approximated)
- (A × B) captures the essential changes needed
- Much smaller than full matrix

---

## Visual Example

### Full Fine-Tuning:
```
Input (512-dim)
      ↓
[ W: 512×512 ] ← All 262K params change
      ↓
Output (512-dim)

Storage: 365MB full model
```

### LoRA Fine-Tuning:
```
Input (512-dim)
      ↓
      ├─→ [ W: 512×512 ] ← FROZEN (no change)
      │         ↓
      │    (Original path)
      │
      └─→ [ B: 16×512 ] ← Trainable (8K params)
                ↓
          [ A: 512×16 ] ← Trainable (8K params)
                ↓
          (Adapter path)
      
Output = Original + Adapter

Storage: 365MB base + 10MB adapter = 375MB total
But adapter is separate file!
```

---

## Mathematical Details (Optional)

### Traditional Update:
```python
# Full fine-tuning
W_original = [512, 512]  # 262K params
ΔW = gradient_update()   # 262K params change
W_new = W_original + ΔW

# Forward pass
output = W_new @ input
```

### LoRA Update:
```python
# LoRA
W = [512, 512]  # FROZEN (no gradient)
A = [512, 16]   # Trainable
B = [16, 512]   # Trainable

# Forward pass
output = W @ input + (A @ B) @ input
#        ↑           ↑
#     original    adapter (low-rank)

# Key: rank(A @ B) = 16, much smaller than rank(W) = 512
```

**Rank explanation:**
- Full matrix W has rank 512 (complex, high-dimensional)
- Adapter A×B has rank 16 (simple, low-dimensional)
- This works because most updates are "simple" (low-rank)

**Think of it like:**
- W captures general knowledge (complex, 512 dimensions)
- A×B captures domain-specific adjustment (simple, 16 dimensions)

---

## Concrete Example: CLIP with LoRA

### Without LoRA (Traditional):

```python
# Load base CLIP
model = open_clip.create_model('EVA02-B-16')  # 149.7M params

# Fine-tune ALL parameters
for param in model.parameters():
    param.requires_grad = True  # All 149.7M params trainable

# Train on vehicles
train(model, vehicle_data)

# Save entire model
torch.save(model.state_dict(), 'model.pt')  # 365MB

# Problem: Original knowledge might be lost!
```

### With LoRA:

```python
from peft import LoraConfig, get_peft_model
import open_clip

# 1. Load base CLIP
base_model = open_clip.create_model('EVA02-B-16')  # 149.7M params

# 2. Freeze ALL base parameters
for param in base_model.parameters():
    param.requires_grad = False  # Nothing changes in base!

# 3. Add LoRA adapters
lora_config = LoraConfig(
    r=16,  # Rank (16 is good default)
    lora_alpha=32,  # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Which layers
    lora_dropout=0.1,
)

model = get_peft_model(base_model, lora_config)

# 4. Check what's trainable
model.print_trainable_parameters()
# Output: trainable params: 2,359,296 / 149,700,000 (1.58%)
#         Only 2.3M params trainable vs 149.7M!

# 5. Train ONLY the adapters
train(model, vehicle_data)  # Only LoRA weights update

# 6. Save ONLY the adapter (tiny!)
model.save_pretrained('adapters/vehicles/')
# Saves: adapter_model.bin (10MB) + adapter_config.json
# Base model untouched!
```

---

## Where LoRA is Added in CLIP

**CLIP has attention layers with weight matrices:**

```
Attention mechanism:
  Q = W_q @ input  (Query)
  K = W_k @ input  (Key)
  V = W_v @ input  (Value)
  O = W_o @ output (Output projection)
```

**LoRA adds adapters to these matrices:**

```
Original:
  Q = W_q @ input

With LoRA:
  Q = (W_q + A_q @ B_q) @ input
       ↑     ↑
     frozen  adapter
```

**Typical target modules:**
- `q_proj` (Query projection)
- `v_proj` (Value projection)
- `k_proj` (Key projection)  
- `o_proj` (Output projection)
- Sometimes: `mlp` layers

**You choose which layers to adapt!**

---

## LoRA Hyperparameters

### 1. Rank (r)

**What it is:** Dimension of the low-rank matrices

```python
r = 16  # Common default

# Creates:
A = [512, 16]  # 8K params
B = [16, 512]  # 8K params
Total: 16K params per layer
```

**Choosing rank:**
- `r=4`: Very small, fast, less expressive (4K params)
- `r=8`: Small, good for simple tasks (8K params)
- `r=16`: **Recommended default** (16K params)
- `r=32`: Larger, more expressive (32K params)
- `r=64`: Large, closer to full fine-tuning (64K params)

**Rule of thumb:**
- Simple domain shift (same task, new data): r=8
- New domain (vehicles, faces, etc.): r=16
- Complex new capability: r=32

---

### 2. Alpha (lora_alpha)

**What it is:** Scaling factor for adapter

```python
lora_alpha = 32  # Typically 2× rank

# Scaling applied:
output = W @ input + (lora_alpha / r) × (A @ B) @ input
                      ↑
                   scaling factor
```

**Choosing alpha:**
- `alpha = r`: Adapter has equal weight to base
- `alpha = 2r`: **Common default** (adapter emphasized)
- `alpha = 4r`: Strong adapter influence

**For your use case:** `alpha = 2 × r` (e.g., r=16, alpha=32)

---

### 3. Target Modules

**Which layers to add adapters to:**

```python
# Option 1: Query and Value only (minimal)
target_modules = ["q_proj", "v_proj"]  # ~2M params

# Option 2: All attention (recommended)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # ~4M params

# Option 3: Attention + MLP (maximum)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "mlp"]  # ~8M params
```

**Trade-offs:**
- More modules → More expressiveness → Larger adapter
- Fewer modules → Faster, smaller → Less flexible

**Recommended:** All attention layers (q/k/v/o)

---

### 4. Dropout

**Regularization to prevent overfitting:**

```python
lora_dropout = 0.1  # 10% dropout on adapter layers
```

**Choosing dropout:**
- Small dataset (<10K): 0.1-0.2 (prevent overfit)
- Medium dataset (10K-100K): 0.05-0.1
- Large dataset (>100K): 0.0-0.05

---

## Using Multiple LoRA Adapters

### Setup: One Base + Multiple Adapters

```
models/
├── eva02_base.pt              # 365MB (shared, frozen)
└── adapters/
    ├── vehicles/
    │   ├── adapter_model.bin   # 10MB
    │   └── adapter_config.json
    ├── people/
    │   ├── adapter_model.bin   # 10MB
    │   └── adapter_config.json
    └── actions/
        ├── adapter_model.bin   # 10MB
        └── adapter_config.json

Total: 365MB + 30MB = 395MB
  vs
3 full models: 365MB × 3 = 1,095MB
```

### Loading Different Adapters:

```python
from peft import PeftModel
import open_clip

# Load base model (once)
base_model = open_clip.create_model('EVA02-B-16')

# Query 1: "blue tesla"
model = PeftModel.from_pretrained(base_model, 'adapters/vehicles/')
result = search(model, "blue tesla")

# Query 2: "person in red jacket"
model = PeftModel.from_pretrained(base_model, 'adapters/people/')
result = search(model, "person in red jacket")

# Query 3: "person smoking" (general - no adapter)
result = search(base_model, "person smoking")
```

### Auto-Select Adapter:

```python
def smart_search(query):
    # Parse query to detect domain
    if any(word in query.lower() for word in ['car', 'tesla', 'toyota', 'vehicle']):
        adapter_path = 'adapters/vehicles/'
    elif any(word in query.lower() for word in ['person', 'man', 'woman', 'jacket']):
        adapter_path = 'adapters/people/'
    elif any(word in query.lower() for word in ['smoking', 'fighting', 'running']):
        adapter_path = 'adapters/actions/'
    else:
        adapter_path = None  # Use base model
    
    # Load model with adapter
    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
    
    return search(model, query)

# Usage
results = smart_search("blue tesla")        # Uses vehicles adapter
results = smart_search("person in jacket")  # Uses people adapter
results = smart_search("sunny day")         # Uses base model
```

---

## Training a LoRA Adapter

### Full Script Example:

```python
#!/usr/bin/env python3
"""
Train LoRA adapter for CLIP
"""

import torch
from peft import LoraConfig, get_peft_model
import open_clip
from torch.utils.data import DataLoader

# 1. Load base CLIP model
print("Loading base model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'EVA02-B-16',
    pretrained='merged2b_s8b_b131k',
    device='cuda'
)
tokenizer = open_clip.get_tokenizer('EVA02-B-16')

# 2. Configure LoRA
print("Adding LoRA adapters...")
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Scaling
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
)

# 3. Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 2,359,296 / 149,700,000 (1.58%)

# 4. Prepare dataset
train_dataset = VehicleDataset(
    data_csv='vehicle_data.csv',
    preprocess=preprocess,
    tokenizer=tokenizer
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 5. Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # Can use higher LR!
loss_fn = torch.nn.CrossEntropyLoss()

# 6. Training loop
print("Training...")
model.train()
for epoch in range(5):
    for images, texts in train_loader:
        images = images.cuda()
        texts = texts.cuda()
        
        # Forward pass
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        
        # Compute CLIP loss
        logits = image_features @ text_features.T
        labels = torch.arange(len(images)).cuda()
        loss = loss_fn(logits, labels)
        
        # Backward pass (only adapter weights update!)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 7. Save ONLY the adapter
print("Saving adapter...")
model.save_pretrained('adapters/vehicles/')
print("✓ Adapter saved (10MB)")
print("✓ Base model unchanged (365MB)")
```

---

## Advantages of LoRA

### 1. Efficiency

**Parameters:**
- Full fine-tuning: 149.7M params trainable
- LoRA: 2.3M params trainable (65× less!)

**Memory:**
- Full fine-tuning: ~8GB GPU memory
- LoRA: ~2GB GPU memory (4× less!)

**Training speed:**
- Full fine-tuning: 100%
- LoRA: 3× faster (fewer params to update)

**Storage:**
- Full fine-tuning: 365MB per model
- LoRA: 365MB base + 10MB per adapter

### 2. Preservation

**Base model never changes:**
- ✅ Original knowledge 100% preserved
- ✅ No catastrophic forgetting
- ✅ Can always fall back to base

### 3. Modularity

**Multiple adapters:**
- ✅ Vehicles: 10MB
- ✅ People: 10MB
- ✅ Actions: 10MB
- ✅ Switch instantly

**Composition:**
- Can even combine adapters!
- `model = base + adapter_vehicles + adapter_people`

### 4. Experimentation

**Easy to try different approaches:**
```bash
# Train multiple adapters in parallel
python train_lora.py --domain vehicles --r 8
python train_lora.py --domain vehicles --r 16
python train_lora.py --domain vehicles --r 32

# Compare results
python eval.py --adapter adapters/r8/
python eval.py --adapter adapters/r16/
python eval.py --adapter adapters/r32/

# Pick best one (all 10MB each)
```

---

## Disadvantages of LoRA

### 1. Slightly Lower Accuracy

**Full fine-tuning:** 90% accuracy  
**LoRA (r=16):** 87-88% accuracy (2-3% lower)  
**LoRA (r=32):** 88-89% accuracy (1-2% lower)

**Trade-off:** Efficiency vs accuracy

### 2. Adapter Switching Overhead

**Loading adapter:** ~50-100ms
- Not free, but fast enough for most use cases

### 3. Need Adapter Selection Logic

**Must decide which adapter to use:**
- Query parsing
- Or train one multi-domain adapter
- Or use base model only

---

## LoRA vs Alternatives

| Method | Trainable Params | Storage | Accuracy | Forgetting Risk |
|--------|-----------------|---------|----------|-----------------|
| **Full fine-tuning** | 100% (149.7M) | 365MB/model | 100% (best) | ⚠️ High |
| **LoRA (r=16)** | 1.5% (2.3M) | 10MB/adapter | 97% | ✅ None |
| **Prefix tuning** | 0.1% (150K) | 1MB/task | 90% | ✅ None |
| **Adapter layers** | 3% (4.5M) | 20MB/task | 98% | ✅ None |

**LoRA is the sweet spot:** Good accuracy, small size, no forgetting

---

## Recommended for Your System

### Strategy: LoRA Adapters

```bash
# Base model (shared)
eva02_base.pt                 # 365MB

# Adapters (specific domains)
adapters/vehicles.pt          # 10MB (tesla, toyota, brands)
adapters/people_clothing.pt   # 10MB (jacket colors, accessories)
adapters/surveillance.pt      # 10MB (angles, lighting, scenes)

Total: 395MB
```

### Implementation:

```python
# Load once at startup
base_model = load_clip('eva02_base.pt')

# Auto-select adapter per query
def search(query):
    adapter = detect_domain(query)  # vehicles, people, or None
    if adapter:
        model = apply_adapter(base_model, adapter)
    else:
        model = base_model
    return semantic_search(model, query)
```

**Benefits for you:**
- ✅ One base model (general knowledge preserved)
- ✅ Small adapters (10MB each, easy to update)
- ✅ Flexible (add new domains without retraining base)
- ✅ Efficient (fast training, low memory)

---

## Getting Started

### Installation:

```bash
pip install peft  # HuggingFace PEFT library (Parameter-Efficient Fine-Tuning)
```

### Quick Test:

```python
from peft import LoraConfig, get_peft_model
import open_clip

# Load model
model = open_clip.create_model('EVA02-B-16')

# Add LoRA
config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, config)

# Check
model.print_trainable_parameters()
# trainable params: 2,359,296 / 149,700,000 (1.58%) ✓
```

---

## Summary

**LoRA in one sentence:**  
Keep base model frozen, add tiny trainable matrices that capture domain-specific knowledge.

**Key benefits:**
- ✅ 65× fewer parameters to train
- ✅ 4× less GPU memory
- ✅ No catastrophic forgetting
- ✅ 10MB adapters vs 365MB models
- ✅ Can have multiple adapters

**Best for:**
- Multiple specialized domains (vehicles, people, actions)
- Preserving base model
- Resource-constrained deployment (Jetson)
- Experimentation

---

## References

- **LoRA Paper:** https://arxiv.org/abs/2106.09685
- **PEFT Library:** https://github.com/huggingface/peft
- **LoRA Tutorial:** https://huggingface.co/docs/peft/conceptual_guides/lora

---

**Last Updated:** 2026-05-13  
**Contact:** chinghokuk@gmail.com
