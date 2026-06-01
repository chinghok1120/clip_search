# Single Model vs Multi-Model Architecture

How to fine-tune CLIP without losing general knowledge.

---

## The Problem

**If we fine-tune CLIP on vehicles:**

```
Base CLIP:
  "person in red jacket" → ✅ Works
  "blue tesla" → ❌ Doesn't work (returns blue things)

Fine-tuned on vehicles:
  "person in red jacket" → ❌ Might degrade! (catastrophic forgetting)
  "blue tesla" → ✅ Works great

Result: Need 2 models = complexity!
```

**This is called "catastrophic forgetting"** - when fine-tuning on new data makes model forget old knowledge.

---

## Solution 1: Careful Fine-Tuning (Keep General Knowledge) ⭐

**Don't fully retrain - do gentle fine-tuning:**

### Technique 1: Low Learning Rate
```python
# BAD: Aggressive fine-tuning (forgets general knowledge)
optimizer = AdamW(model.parameters(), lr=1e-3)  # Too high!
epochs = 20  # Too many!

# GOOD: Gentle fine-tuning (keeps general knowledge)
optimizer = AdamW(model.parameters(), lr=1e-6)  # Very low
epochs = 3-5  # Just enough to learn vehicles
```

**Why it works:**
- Small updates to weights
- Model learns vehicles WITHOUT drastically changing
- General knowledge preserved

---

### Technique 2: Mix General + Vehicle Data

**Train on BOTH at same time:**

```python
# Training data mix
training_data = [
    # 60% vehicle data
    ("tesla_car.jpg", "blue Tesla Model 3"),
    ("toyota_car.jpg", "red Toyota Camry"),
    
    # 40% general surveillance data
    ("person1.jpg", "person in red jacket"),
    ("person2.jpg", "person with backpack"),
    ("scene1.jpg", "crowded parking lot"),
]

# Fine-tune on mixed data
model.train(training_data)
```

**Result:** One model that knows both vehicles AND general concepts

**Data mix strategy:**
- 60% new domain (vehicles)
- 40% general (preserve old knowledge)
- Ratio depends on your priority

---

### Technique 3: Incremental Fine-Tuning

**Start from base CLIP, add knowledge gradually:**

```python
# Stage 1: Base CLIP (already knows general stuff)
model = load_clip("EVA02-B-16")

# Stage 2: Add vehicle knowledge (gentle)
finetune(model, vehicle_data, lr=1e-6, epochs=3)

# Stage 3: Test both
assert model_works_on("person in red jacket")  # Still works ✓
assert model_works_on("blue tesla")  # Now works ✓
```

**Testing after fine-tuning:**
```bash
# Test general queries still work
python test_model.py --queries "person in red jacket,man with backpack,crowded street"

# Test vehicle queries improved
python test_model.py --queries "blue tesla,red toyota,white bmw"

# Both should work!
```

---

## Solution 2: LoRA Adapters (Best of Both Worlds) ⭐⭐

**Keep base model frozen, add tiny trainable adapters:**

### What is LoRA?

**LoRA (Low-Rank Adaptation):**
- Base CLIP stays frozen (365MB)
- Add small adapter layers (5-10MB)
- Train ONLY the adapters
- Base knowledge preserved 100%

**Architecture:**
```
                Base CLIP (frozen)
                     ↓
Input → [LoRA Adapter: Vehicles] → Output
         ↓
    Only this trains (10MB)
    Base CLIP unchanged
```

**Advantages:**
- ✅ Base model never changes (general knowledge preserved)
- ✅ Adapter is tiny (5-10MB vs 365MB full model)
- ✅ Can have multiple adapters:
  - `adapter_vehicles.pt` (10MB)
  - `adapter_people.pt` (10MB)
  - `adapter_actions.pt` (10MB)
- ✅ Switch adapters instantly (50ms)

**Storage:**
```
models/
├── eva02_base.pt          # 365MB (shared)
├── adapters/
│   ├── vehicles.pt        # 10MB
│   ├── people.pt          # 10MB
│   └── general.pt         # 10MB
Total: 395MB (vs 1095MB for 3 full models)
```

### Implementation with LoRA:

```python
from peft import LoraConfig, get_peft_model
import open_clip

# 1. Load base CLIP (frozen)
base_model = open_clip.create_model('EVA02-B-16')

# 2. Add LoRA adapter
lora_config = LoraConfig(
    r=16,  # Rank (higher = more capacity)
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.1,
)
model = get_peft_model(base_model, lora_config)

# 3. Train ONLY the adapter (base frozen)
model.print_trainable_parameters()
# Output: trainable params: 2.3M / 149.7M (1.5%)

# 4. Fine-tune on vehicles
train(model, vehicle_data)

# 5. Save adapter (tiny!)
model.save_pretrained("adapters/vehicles/")
# Saves only: adapter_config.json + adapter_model.bin (10MB)
```

### Using adapters:

```python
# Load base model once
base_model = load_clip("EVA02-B-16")

# Query: "blue tesla"
adapter = load_adapter("adapters/vehicles.pt")
model = apply_adapter(base_model, adapter)
result = search(model, "blue tesla")  # Uses vehicle knowledge

# Query: "person in red jacket"
adapter = load_adapter("adapters/people.pt")
model = apply_adapter(base_model, adapter)
result = search(model, "person in red jacket")  # Uses people knowledge
```

**Auto-detect which adapter to use:**
```python
def smart_search(query):
    # Detect query type
    if is_vehicle_query(query):  # "tesla", "toyota", "car"
        adapter = "vehicles"
    elif is_person_query(query):  # "person", "jacket", "wearing"
        adapter = "people"
    else:
        adapter = None  # Use base model
    
    # Search
    model = load_with_adapter(adapter)
    return search(model, query)
```

---

## Solution 3: Multi-Domain Fine-Tuning (One Model for Everything)

**Fine-tune on ALL surveillance domains at once:**

```python
training_data = [
    # Vehicles (30%)
    ("car1.jpg", "blue Tesla Model 3"),
    ("car2.jpg", "red Toyota Camry"),
    
    # People (40%)
    ("person1.jpg", "person in red jacket"),
    ("person2.jpg", "man with backpack"),
    
    # Scenes (20%)
    ("scene1.jpg", "crowded parking lot"),
    ("scene2.jpg", "empty hallway"),
    
    # Actions (10%)
    ("action1.jpg", "person walking"),
    ("action2.jpg", "person sitting"),
]

# Fine-tune on ALL domains
model = finetune_clip(base_clip, training_data)
```

**Result:** One "surveillance-optimized CLIP"
- ✅ Knows vehicles (tesla, toyota, bmw)
- ✅ Knows people (clothing, accessories)
- ✅ Knows scenes (parking lot, entrance, hallway)
- ✅ Optimized for surveillance angles/lighting

**This is the best production approach** - one model, optimized for your entire use case.

---

## Comparison

| Approach | # Models | Storage | Accuracy (Vehicles) | Accuracy (General) | Complexity |
|----------|----------|---------|-------------------|-------------------|------------|
| **Base CLIP only** | 1 | 365MB | ❌ Poor (30%) | ✅ Good (85%) | Low |
| **Two separate models** | 2 | 730MB | ✅ Great (90%) | ✅ Good (85%) | High (routing) |
| **Gentle fine-tuning** | 1 | 365MB | ✅ Good (75%) | ✅ Good (80%) | Low |
| **LoRA adapters** | 1 base + 3 adapters | 395MB | ✅ Great (88%) | ✅ Great (85%) | Medium |
| **Multi-domain fine-tuning** | 1 | 365MB | ✅ Great (85%) | ✅ Great (88%) | Low |

---

## Recommended Approach

### For Your System: Multi-Domain Fine-Tuning ⭐

**Why:**
- ✅ One model (simple deployment)
- ✅ Optimized for ALL your queries (vehicles + people + scenes)
- ✅ Adapts to surveillance domain (angles, lighting, resolution)
- ✅ No query routing needed

**Implementation:**

```bash
# 1. Collect diverse surveillance data
# - Vehicles: 10K images (CompCars + your footage)
# - People: 10K images (CrowdHuman + your footage)
# - Scenes: 5K images (your footage)

# 2. Prepare mixed training data
python prepare_surveillance_dataset.py \
  --vehicles ~/datasets/compcars \
  --people ~/datasets/crowdhuman \
  --scenes ~/datasets/surveillance_scenes \
  --output surveillance_training.csv

# 3. Fine-tune on mixed data
python finetune_clip.py \
  --data surveillance_training.csv \
  --base-model EVA02-B-16 \
  --lr 1e-6 \
  --epochs 5 \
  --output models/eva02_surveillance.pt

# 4. Test on both domains
python test_model.py \
  --model models/eva02_surveillance.pt \
  --test-general "person in red jacket,man with backpack" \
  --test-vehicles "blue tesla,red toyota"

# 5. Deploy single model
cp models/eva02_surveillance.pt production/
```

**Training data composition:**
- 40% People/clothing (most common queries)
- 30% Vehicles (important domain)
- 20% Scenes/objects (context)
- 10% Actions (bonus)

---

## Alternative: LoRA for Maximum Flexibility

**If you want to keep experimenting:**

```bash
# Train separate LoRA adapters
python train_lora.py --domain vehicles --output adapters/vehicles.pt
python train_lora.py --domain people --output adapters/people.pt
python train_lora.py --domain actions --output adapters/actions.pt

# Deploy base + adapters
models/
├── eva02_base.pt       # 365MB (shared)
└── adapters/
    ├── vehicles.pt     # 10MB
    ├── people.pt       # 10MB
    └── actions.pt      # 10MB

# Auto-select adapter per query
python search.py --query "blue tesla" --auto-adapter
```

**Pros:**
- Can add new domains without retraining base
- Small storage footprint
- Mix and match adapters

**Cons:**
- Need adapter routing logic
- Slightly more complex

---

## Catastrophic Forgetting Prevention

**Test checklist after fine-tuning:**

```python
# Before fine-tuning: Base CLIP
test_queries = [
    "person in red jacket",      # 85% accuracy
    "blue car",                  # 30% accuracy (poor on brands)
    "crowded parking lot",       # 75% accuracy
]

# After fine-tuning: Check preservation
test_queries_after = [
    "person in red jacket",      # Should still be ~80%+ ✓
    "blue tesla",                # Should be 85%+ now ✓
    "crowded parking lot",       # Should still be ~70%+ ✓
]

# If general queries drop >10%, fine-tuning was too aggressive
```

**Prevention techniques:**
1. ✅ Low learning rate (1e-6)
2. ✅ Few epochs (3-5)
3. ✅ Mix general data in training
4. ✅ Early stopping (monitor general queries)
5. ✅ LoRA (base never changes)

---

## Summary

**You asked: "Will I need 2 models?"**

**Answer: No! Use ONE of these approaches:**

1. **Multi-domain fine-tuning** (Best for production)
   - Train on vehicles + people + scenes together
   - One model, handles everything
   - Optimized for your surveillance domain

2. **LoRA adapters** (Best for flexibility)
   - One base model + tiny adapters
   - Can add new domains easily
   - Slightly more complex

3. **Gentle fine-tuning** (Quick start)
   - Very low learning rate
   - Mix vehicle + general data
   - One model, minimal forgetting

**DON'T: Maintain two separate models** (too complex)

---

## Next Steps

```bash
# Week 1: Test gentle fine-tuning
python finetune_clip.py \
  --data mixed_data.csv \
  --lr 1e-6 \
  --epochs 3

# Week 2: If works well, collect more data
# Week 3: Full multi-domain fine-tuning
# Deploy: Single surveillance-optimized model
```

---

**Last Updated:** 2026-05-13  
**Contact:** chinghokuk@gmail.com
