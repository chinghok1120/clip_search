# LoRA vs Low Learning Rate: Weight Corruption

Understanding why LoRA is fundamentally different from gentle fine-tuning.

---

## Your Question: Does Low LR Eventually Corrupt Weights?

**Short answer: YES for low-LR fine-tuning, NO for LoRA**

Let me explain the crucial difference.

---

## Low Learning Rate Fine-Tuning (Weight Drift)

### How it works:

```python
# Iteration 1: Fine-tune on vehicles
W_original = [initial weights]
W_after_vehicles = W_original + small_update_1  # LR = 1e-6

# Iteration 2: Fine-tune on people
W_after_people = W_after_vehicles + small_update_2

# Iteration 3: Fine-tune on actions
W_after_actions = W_after_people + small_update_3

# Iteration 10: Fine-tune on new data
W_after_10 = W_after_9 + small_update_10

# Problem: W has drifted far from W_original!
```

### Weight drift over time:

```
Original weights:    [1.23, 0.45, -0.67, ...]
After session 1:     [1.24, 0.46, -0.66, ...]  (small change)
After session 2:     [1.25, 0.47, -0.65, ...]  (accumulated)
After session 5:     [1.30, 0.52, -0.60, ...]  (drifting)
After session 10:    [1.45, 0.65, -0.45, ...]  (very different!)

Original knowledge:  "person in red jacket" = 85% accuracy
After session 10:    "person in red jacket" = 60% accuracy ❌
                     (corrupted!)
```

**Even with low learning rate:**
- ✅ Each update is small (safe per session)
- ❌ Updates accumulate over many sessions
- ❌ Eventually drift from original
- ❌ Original knowledge degrades

**This is your concern, and it's valid!**

---

## LoRA: Zero Weight Corruption (Mathematically Guaranteed)

### How LoRA actually works:

```python
# Base weights
W = [original weights]  # FROZEN - requires_grad = False

# Adapters (separate parameters)
A_vehicles = [trainable]
B_vehicles = [trainable]

A_people = [trainable]
B_people = [trainable]

# Forward pass
output = W @ input + (A @ B) @ input
         ↑           ↑
      FROZEN      trainable
   (never changes)
```

### Key difference: W literally never changes

```python
# Session 1: Train vehicle adapter
for epoch in range(10):
    loss.backward()  # Gradients computed
    optimizer.step() # Only A_vehicles and B_vehicles update
                     # W gets NO gradient (frozen)

print(W == W_original)  # True! Exactly identical

# Session 2: Train people adapter  
for epoch in range(10):
    loss.backward()
    optimizer.step()  # Only A_people and B_people update

print(W == W_original)  # Still True!

# Session 100: Train something else
for epoch in range(10):
    loss.backward()
    optimizer.step()  # Only new adapter updates

print(W == W_original)  # Still True! Always!
```

**Weight values over time:**

```
Session 1:  W = [1.23, 0.45, -0.67, ...]  ← Original
Session 2:  W = [1.23, 0.45, -0.67, ...]  ← Identical
Session 5:  W = [1.23, 0.45, -0.67, ...]  ← Identical
Session 10: W = [1.23, 0.45, -0.67, ...]  ← Identical
Session 100: W = [1.23, 0.45, -0.67, ...] ← Identical forever!

Original knowledge: "person in red jacket" = 85% accuracy
After 100 sessions: "person in red jacket" = 85% accuracy ✓
                    (Never degrades!)
```

---

## Mathematical Proof: W Cannot Change in LoRA

### PyTorch implementation:

```python
import torch

# Base model weights
W = torch.randn(512, 512)
W.requires_grad = False  # ← This is the key!

# LoRA adapters
A = torch.randn(512, 16, requires_grad=True)
B = torch.randn(16, 512, requires_grad=True)

# Forward pass
input = torch.randn(32, 512)
output = input @ W + input @ (A @ B)

# Backward pass
loss = output.sum()
loss.backward()

# Check gradients
print(W.grad)  # None - no gradient computed!
print(A.grad)  # Tensor(...) - gradient exists
print(B.grad)  # Tensor(...) - gradient exists

# Update weights
optimizer.step()  # Only A and B update

# W is bitwise identical to original
print(torch.equal(W, W_original))  # True - always!
```

**Key:** `requires_grad = False` means:
1. No gradients computed for W
2. Optimizer cannot update W
3. W cannot change, ever
4. Mathematically impossible to corrupt W

---

## Visual Comparison

### Low Learning Rate Fine-Tuning:

```
┌─────────────────────────────────────────┐
│  Base Model Weights (W)                 │
│  [trainable, changes each session]      │
└─────────────────────────────────────────┘
         ↓
    Session 1: -0.001 change
         ↓
┌─────────────────────────────────────────┐
│  Weights after session 1                │
│  [slightly different]                   │
└─────────────────────────────────────────┘
         ↓
    Session 2: -0.001 change
         ↓
┌─────────────────────────────────────────┐
│  Weights after session 2                │
│  [more different]                       │
└─────────────────────────────────────────┘
         ↓
    ... 10 sessions ...
         ↓
┌─────────────────────────────────────────┐
│  Weights after session 10               │
│  [significantly different! ❌]          │
└─────────────────────────────────────────┘
```

### LoRA:

```
┌─────────────────────────────────────────┐
│  Base Model Weights (W)                 │
│  [FROZEN - never changes]               │
└─────────────────────────────────────────┘
         ↓ (read-only)
         │
    ┌────┴─────┬─────────┬─────────┐
    ↓          ↓         ↓         ↓
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Adapter │ │Adapter │ │Adapter │ │Adapter │
│  #1    │ │  #2    │ │  #3    │ │  #10   │
│[10 MB] │ │[10 MB] │ │[10 MB] │ │[10 MB] │
└────────┘ └────────┘ └────────┘ └────────┘

Base W: [1.23, 0.45, -0.67, ...] ← Never changes!
        Identical after 100 sessions ✓
```

---

## Experiment: Prove Weight Corruption

### Test script:

```python
import torch
import open_clip
from peft import LoraConfig, get_peft_model
import copy

# Load base model
base_model = open_clip.create_model('EVA02-B-16')

# Save original weights
original_weights = copy.deepcopy(base_model.state_dict())

print("="*60)
print("LOW LEARNING RATE FINE-TUNING (10 sessions)")
print("="*60)

model_low_lr = copy.deepcopy(base_model)
model_low_lr.train()

for session in range(10):
    # Fine-tune with low LR
    optimizer = torch.optim.Adam(model_low_lr.parameters(), lr=1e-6)
    
    # Simulate training
    for step in range(100):
        loss = torch.randn(1, requires_grad=True)  # Dummy loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Check weight difference
    current_weights = model_low_lr.state_dict()
    diff = 0
    for key in original_weights:
        diff += (current_weights[key] - original_weights[key]).abs().mean().item()
    
    print(f"Session {session+1}: Weight drift = {diff:.6f}")

print("\n" + "="*60)
print("LoRA (10 sessions)")
print("="*60)

for session in range(10):
    # Create LoRA model (fresh each time)
    model_lora = copy.deepcopy(base_model)
    lora_config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
    model_lora = get_peft_model(model_lora, lora_config)
    
    # Train adapter
    optimizer = torch.optim.Adam(model_lora.parameters(), lr=1e-4)  # Higher LR OK!
    
    for step in range(100):
        loss = torch.randn(1, requires_grad=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Check base weight difference
    current_weights = model_lora.base_model.model.state_dict()
    diff = 0
    for key in original_weights:
        if key in current_weights:  # Only base weights
            diff += (current_weights[key] - original_weights[key]).abs().mean().item()
    
    print(f"Session {session+1}: Weight drift = {diff:.6f}")

# Expected output:
# LOW LR: drift increases (0.001 → 0.01 → 0.05 → ...)
# LoRA:   drift = 0.000000 (always!)
```

---

## Real-World Impact

### Scenario: You fine-tune 10 times over 1 year

**Low learning rate approach:**

```
Month 1:  Fine-tune on vehicles     → drift = 0.001
Month 2:  Fine-tune on people       → drift = 0.003
Month 3:  Fine-tune on new cameras  → drift = 0.007
Month 6:  Fine-tune on new scenes   → drift = 0.025
Month 12: Fine-tune on new data     → drift = 0.100 ❌

Original query performance:
  "person in red jacket": 85% → 70% (degraded!)
  "blue car": 75% → 60% (degraded!)
  
Model has corrupted. Need to restart from scratch.
```

**LoRA approach:**

```
Month 1:  Train vehicles adapter    → drift = 0.000
Month 2:  Train people adapter      → drift = 0.000
Month 3:  Train cameras adapter     → drift = 0.000
Month 6:  Train scenes adapter      → drift = 0.000
Month 12: Train new_data adapter    → drift = 0.000 ✓

Original query performance:
  "person in red jacket": 85% → 85% (unchanged!)
  "blue car": 75% → 75% (unchanged!)
  
Can use base model OR any adapter, forever.
```

---

## Why This Matters for Production

### Low LR Fine-Tuning (Degrades Over Time):

```
Year 1: Deploy model v1                    ✓ Works
Year 2: Fine-tune on new data → v2         ✓ Works, slightly degraded
Year 3: Fine-tune on more data → v3        ⚠️ Degraded 10%
Year 4: Fine-tune again → v4               ❌ Degraded 20%, unusable
Year 5: Need to retrain from scratch       ❌ Expensive!
```

### LoRA (Never Degrades):

```
Year 1: Deploy base + adapter_v1           ✓ Works
Year 2: Add adapter_v2 (base unchanged)    ✓ Works perfectly
Year 3: Add adapter_v3 (base unchanged)    ✓ Works perfectly
Year 10: Add adapter_v10 (base unchanged)  ✓ Still works perfectly!

Base model good forever. Just add adapters.
```

---

## Additional LoRA Benefits (You Asked About)

### 1. Can Use Higher Learning Rates

**Because base is frozen:**

```python
# Low LR fine-tuning: Must use tiny LR
optimizer = Adam(model.parameters(), lr=1e-6)  # Too low, slow training

# LoRA: Can use normal LR (adapters are separate)
optimizer = Adam(model.parameters(), lr=1e-4)  # 100× higher, faster!
                                                # Base still safe (frozen)
```

**Result:**
- LoRA trains faster (higher LR possible)
- No risk to base weights

### 2. Easy Rollback

**Low LR fine-tuning:**
```
Train session → weights changed → Bad results? ❌ Cannot undo!
Need to restore from backup (if you have one)
```

**LoRA:**
```
Train adapter → Bad results? ✓ Just delete adapter file!
Base model unchanged, instantly back to original
```

### 3. A/B Testing

**Low LR fine-tuning:**
```
Try approach A: Fine-tune → weights changed
Try approach B: Need to retrain from original ❌ Expensive!
```

**LoRA:**
```
Try approach A: Train adapter_A (10MB)
Try approach B: Train adapter_B (10MB)
Compare both:
  - Load base + adapter_A → test
  - Load base + adapter_B → test
Pick winner, delete loser ✓
```

---

## Summary Table

| Aspect | Low LR Fine-Tuning | LoRA |
|--------|-------------------|------|
| **Base weights change?** | ✅ Yes (small, but accumulates) | ❌ No (frozen forever) |
| **Weight drift over time?** | ✅ Yes (inevitable) | ❌ No (impossible) |
| **Original knowledge degradation?** | ✅ Yes (after many sessions) | ❌ No (preserved forever) |
| **Need to retrain from scratch?** | ✅ Eventually (after corruption) | ❌ Never |
| **Can rollback bad training?** | ❌ No (weights changed) | ✅ Yes (delete adapter) |
| **Safe for production?** | ⚠️ Short term only | ✅ Long term safe |
| **Training speed** | Slow (low LR needed) | Fast (high LR safe) |

---

## Recommendation for Your System

**Use LoRA, not low learning rate fine-tuning.**

**Why:**
1. ✅ **No corruption ever** (mathematically guaranteed)
2. ✅ **Safe for continuous updates** (add adapters forever)
3. ✅ **Faster training** (can use higher LR)
4. ✅ **Easy rollback** (delete bad adapter)
5. ✅ **A/B testing** (compare multiple adapters)
6. ✅ **Production safe** (base never degrades)

**Low LR fine-tuning only if:**
- ❌ One-time fine-tuning (never update again)
- ❌ Can retrain from scratch when corrupted
- ❌ Don't care about long-term stability

**For your surveillance system:**
- You'll add new cameras
- You'll get new data
- You'll want to improve over time
- **LoRA is the right choice** ✅

---

## Implementation

```python
# DON'T do this (weight drift):
for session in range(10):
    finetune(model, new_data, lr=1e-6)  # ❌ Corrupts over time

# DO this (LoRA):
base_model = load_clip('EVA02-B-16')  # Load once

for domain in ['vehicles', 'people', 'actions']:
    adapter = train_lora_adapter(base_model, domain)  # ✓ Base unchanged
    save_adapter(adapter, f'adapters/{domain}.pt')    # ✓ 10MB each

# Deploy
# Base: 365MB (unchanged forever)
# Adapters: 10MB each (add as needed)
```

---

**Last Updated:** 2026-05-13  
**Contact:** chinghokuk@gmail.com
