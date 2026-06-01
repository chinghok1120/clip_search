# Vision Model Landscape 2026: Beyond EVA-02

Understanding different vision model architectures and what's best for surveillance search.

---

## Model Categories

### 1. CLIP-Style Models (What We Use)

**Architecture:** Dual-encoder (image encoder + text encoder)

```
Image → Vision Encoder → Image Embedding (512-dim)
Text  → Text Encoder   → Text Embedding (512-dim)
                          ↓
                    Cosine Similarity
```

**Characteristics:**
- ✅ **Fast**: Direct embedding comparison (~1-2ms)
- ✅ **Efficient**: Precompute image embeddings once
- ✅ **Scalable**: FAISS can search millions of embeddings
- ❌ **Limited reasoning**: Just similarity, no understanding

**Models in this category:**
| Model | Year | Parameters | Performance | Notes |
|-------|------|-----------|-------------|-------|
| **CLIP ViT-B** | 2021 | 86M | Baseline | Original, widely used |
| **OpenCLIP ViT-L** | 2022 | 428M | +15% vs CLIP | Better training |
| **EVA-02-B** (current) | 2023 | 150M | +20% vs CLIP | What we use now |
| **EVA-02-L** | 2023 | 428M | +25% vs CLIP | Larger version |
| **SigLIP-L** | 2023 | 428M | +30% vs CLIP | Google, better loss |
| **MetaCLIP-L** | 2024 | 428M | +32% vs CLIP | Meta, better data curation |
| **EVA-CLIP-8B** | 2024 | 5B | +40% vs CLIP | State-of-art, huge |
| **DFN-CLIP** | 2024 | 428M | +35% vs CLIP | Better fine-grained details |

**Best for:** Fast similarity search, real-time systems, millions of images

---

### 2. Vision-Language Models (VLMs) - ChatGPT Approach

**Architecture:** Vision encoder + Large Language Model

```
Image → Vision Encoder → Visual Tokens
                          ↓
Text Query → [Visual Tokens + Text Tokens] → LLM → Generated Answer
```

**How ChatGPT/Claude/DeepSeek work:**
1. **Vision Encoder**: Converts image to visual tokens (like CLIP)
2. **LLM Decoder**: Processes visual tokens + text together
3. **Generation**: Produces text descriptions, answers questions
4. **Reasoning**: Can understand context, actions, relationships

**Example:**
```
CLIP approach:
Query: "person smoking cigarette"
→ Compute similarity → Return top matches

VLM approach:
Query: "Is this person smoking a cigarette?"
→ LLM analyzes image → "Yes, there is a person holding a cigarette near their mouth"
```

**Models in this category:**

| Model | Year | Size | Capabilities | Access |
|-------|------|------|--------------|--------|
| **GPT-4V** | 2023 | Unknown | Image understanding, OCR, reasoning | API only ($) |
| **Claude 3 Opus** | 2024 | Unknown | Image analysis, document understanding | API only ($) |
| **Gemini Pro Vision** | 2024 | Unknown | Video understanding, multimodal | API only ($) |
| **LLaVA-1.6** | 2024 | 7B-34B | Open GPT-4V alternative | Open-source ✅ |
| **Qwen-VL** | 2024 | 7B-72B | Chinese+English, strong reasoning | Open-source ✅ |
| **DeepSeek-VL** | 2024 | 7B | Efficient VLM, good reasoning | Open-source ✅ |
| **CogVLM2** | 2024 | 19B | Strong visual grounding | Open-source ✅ |
| **InternVL-2** | 2024 | 8B-76B | State-of-art open VLM | Open-source ✅ |

**Characteristics:**
- ✅ **Better understanding**: Can reason about actions, context
- ✅ **Fine-grained**: Better at small objects, activities
- ✅ **Natural language**: Can answer "Is this person smoking?"
- ❌ **Slow**: 100-500ms per image (100× slower than CLIP)
- ❌ **Not scalable**: Cannot precompute embeddings
- ❌ **Resource heavy**: 7B-70B parameters (vs 150M for CLIP)

**Best for:** Reranking, verification, complex queries

---

### 3. Hybrid Approach: CLIP + VLM (Best of Both)

**Architecture:**
```
Stage 1: CLIP (Fast filter)
  15,000 images → CLIP search → Top 100 candidates (2ms)

Stage 2: VLM (Accurate verification)
  100 candidates → VLM analyze each → Top 10 verified results (5 seconds)
```

**Example workflow:**
```
User query: "person smoking cigarette"

Step 1: CLIP search
  Query: "person with hand near face"
  → Returns 100 candidate images (CLIP is weak on "smoking")

Step 2: VLM verification
  For each candidate:
    VLM prompt: "Is there a person smoking a cigarette in this image? Answer yes or no."
    → Filter to images where VLM says "yes"

Final result: 10 images, high accuracy
```

**Benefits:**
- ✅ CLIP speed for initial filtering
- ✅ VLM accuracy for verification
- ✅ Best of both worlds

**Drawback:**
- Still slow for verification stage (but only on top-k)

---

## Can We Do Better Than EVA-02?

### Option 1: Upgrade to Newer CLIP Models

**Recommendation: MetaCLIP-L or DFN-CLIP (2024)**

| Model | Params | Accuracy vs EVA-02-B | Speed vs EVA-02-B | Memory | Jetson Compatible |
|-------|--------|---------------------|-------------------|--------|-------------------|
| EVA-02-B (current) | 150M | Baseline | Baseline (21k img/min) | 0.8GB | ✅ Yes |
| EVA-02-L | 428M | +8-12% | 0.6× (13k img/min) | 2GB | ✅ Yes |
| SigLIP-L | 428M | +10-15% | 0.6× (13k img/min) | 2GB | ✅ Yes |
| **MetaCLIP-L** | 428M | +12-18% | 0.6× (13k img/min) | 2GB | ✅ Yes |
| **DFN-CLIP-L** | 428M | +15-20% (fine details) | 0.5× (11k img/min) | 2GB | ✅ Yes |
| EVA-CLIP-8B | 5B | +25-35% | 0.1× (2k img/min) | 20GB | ❌ No (too big) |

**Quick win:** Upgrade to **MetaCLIP-L** or **DFN-CLIP-L**
- Still fast enough (>10k img/min, 10× above target)
- 15-20% better accuracy
- Drop-in replacement (same API as EVA-02)
- Fits on Jetson Orin Nano (2GB VRAM)

**Implementation:**
```python
# Current
model, _, preprocess = open_clip.create_model_and_transforms(
    'EVA02-B-16',
    pretrained='merged2b_s8b_b131k'
)

# Upgrade to MetaCLIP
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14-336',
    pretrained='metaclip_fullcc'  # MetaCLIP weights
)

# Or DFN-CLIP (if available in OpenCLIP)
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14',
    pretrained='dfn_clip'
)
```

---

### Option 2: Use VLM as Reranker

**Architecture:**
```
User query: "person smoking cigarette"
    ↓
Step 1: CLIP search (EVA-02-B)
  → Find top 50 candidates (fast, ~2ms)
    ↓
Step 2: VLM verification (LLaVA-1.6 or DeepSeek-VL)
  For each of 50 images:
    Prompt: "Does this image show a person smoking? Answer yes/no and explain."
  → Keep only images where VLM confirms "yes"
    ↓
Step 3: Return verified results
  → 5-10 high-confidence matches
```

**VLM choices for verification:**

| Model | Size | Speed (per image) | Accuracy | Deployment |
|-------|------|------------------|----------|------------|
| **LLaVA-1.6-7B** | 7B | ~200ms | Good | ✅ Runs on Jetson |
| **DeepSeek-VL-7B** | 7B | ~150ms | Good | ✅ Runs on Jetson |
| **Qwen-VL-7B** | 7B | ~180ms | Good | ✅ Runs on Jetson |
| GPT-4V | Large | ~500ms | Excellent | ❌ API only ($$$) |
| Claude 3 | Large | ~400ms | Excellent | ❌ API only ($$$) |

**Pros:**
- ✅ Dramatically better for action queries ("smoking", "fighting")
- ✅ Can verify small objects
- ✅ Can answer "why" (explanations)
- ✅ Open-source options available

**Cons:**
- ⚠️ Adds 5-10 seconds latency for verification
- ⚠️ Requires running second model (7B params)
- ⚠️ More complex system

**When to use:**
- High-value queries where accuracy matters most
- Actions/activities (smoking, fighting, falling)
- Small object detection (weapons, phones)
- Forensic search (not real-time)

---

### Option 3: Fine-Tune Current Model

**Don't overlook this!** Fine-tuning EVA-02-B on surveillance data often outperforms upgrading to a larger model.

**Comparison:**

| Approach | Accuracy Gain | Cost | Effort | Speed Impact |
|----------|---------------|------|--------|--------------|
| Upgrade EVA-02-B → MetaCLIP-L | +12-18% | Free | Low (code change) | 0.6× slower |
| Fine-tune EVA-02-B on surveillance | +20-40% | $100 | Medium (labeling) | No change |
| Use VLM reranker | +30-50% | $0-500 | High (integration) | +5s latency |

**Why fine-tuning wins:**
- Your specific domain: overhead cameras, lighting, angles
- Your specific queries: uniform types, vehicle types, common scenarios
- Your specific objects: company logos, facility layout

**Example improvement on fine-tuned model:**
```
Query: "delivery person in brown uniform"

Base EVA-02-B: 60% accuracy (confuses with other people)
Fine-tuned EVA-02-B: 85% accuracy (learned "brown uniform" = UPS/FedEx)
```

---

## What ChatGPT/DeepSeek Actually Use

### GPT-4V Architecture (Estimated)
```
Image → CLIP-like Vision Encoder (pretrained)
          ↓
    Visual Tokens (256-1024 tokens)
          ↓
    GPT-4 Transformer (1.7T parameters)
          ↓
    Generated Text Description
```

**Key differences from CLIP:**
1. **Unified model**: Vision and language in one model
2. **Generative**: Produces descriptions, not just embeddings
3. **Reasoning**: Can think step-by-step
4. **Massive scale**: 100× larger than EVA-02-B

**Why they're slow for search:**
- Must process each image individually
- Cannot precompute embeddings (generative, not just encoding)
- 500ms per image × 15,000 images = 2 hours per query!

**Why they're good for verification:**
- Understand nuance and context
- Can detect actions and small objects
- Explain reasoning

---

### DeepSeek-VL Architecture
```
Image → SigLIP Vision Encoder (pretrained, frozen)
          ↓
    Visual Tokens
          ↓
    DeepSeek LLM (7B-67B params)
          ↓
    Generated Answer
```

**Advantages:**
- Open-source (you can deploy locally)
- Smaller than GPT-4V (7B vs ~1T)
- Good reasoning for its size
- Can run on Jetson with quantization

**Use case for your system:**
```python
# Hybrid search example
def search_with_verification(query, top_k=10):
    # Stage 1: CLIP fast search
    clip_results = clip_search(query, top_k=50)  # Get 50 candidates
    
    # Stage 2: VLM verification
    verified_results = []
    for img in clip_results:
        # Ask VLM to verify
        vlm_prompt = f"Does this image match: '{query}'? Answer yes or no."
        answer = deepseek_vl(img, vlm_prompt)
        
        if "yes" in answer.lower():
            verified_results.append(img)
        
        if len(verified_results) >= top_k:
            break
    
    return verified_results
```

---

## Practical Recommendations for Your System

### Immediate (This Week)
**Upgrade to MetaCLIP-L or keep EVA-02-B + fine-tune**

```python
# Option A: Drop-in upgrade (15% better accuracy)
model = open_clip.create_model_and_transforms(
    'ViT-L-14-336',
    pretrained='metaclip_fullcc'
)

# Option B: Keep EVA-02-B, fine-tune later
# (Fine-tuning often better ROI than larger model)
```

**Why:** Simple change, measurable improvement, still fast

---

### Short-term (Next Month)
**Add metadata filtering + query suggestions**

- Time/camera filters (handles temporal queries)
- "Suggested searches" UI (guides users to good queries)
- Log failed searches (identify what to fine-tune for)

**Why:** Complements CLIP's strengths, avoids weaknesses

---

### Medium-term (2-3 Months)
**Fine-tune on surveillance data**

1. Collect 10K labeled surveillance images
2. Fine-tune EVA-02-B or MetaCLIP-L
3. A/B test vs base model
4. Deploy if >20% improvement

**Why:** Best accuracy-per-dollar, domain-specific

---

### Long-term (6 Months)
**Add VLM reranker for high-value queries**

- Deploy LLaVA-1.6-7B or DeepSeek-VL-7B on Jetson
- Use for action queries: smoking, fighting, falling
- CLIP first pass → VLM verification
- Async processing (5-10 second delay OK for forensics)

**Why:** Handles CLIP's weak spots (actions, small objects)

---

## Summary Table

| Approach | Accuracy | Speed | Cost | Complexity | Recommendation |
|----------|----------|-------|------|------------|----------------|
| **Current (EVA-02-B)** | Baseline | Fast (21k/min) | $0 | Low | ✅ Keep for MVP |
| **Upgrade to MetaCLIP-L** | +15% | Fast (13k/min) | $0 | Low | ✅ Easy win |
| **Fine-tune EVA-02-B** | +30% | Fast (21k/min) | $100 | Med | ✅ Best ROI |
| **Add VLM reranker** | +50% | Slow (+5s) | $0 | High | ⏳ Phase 3 |
| **EVA-CLIP-8B** | +35% | Slow (2k/min) | $0 | Med | ❌ Too slow |
| **GPT-4V API** | +60% | Very slow | $$$$ | Low | ❌ Not scalable |

---

## Recommended Path Forward

### Phase 1: Quick Wins (Now)
```bash
# Test MetaCLIP-L vs EVA-02-B
python scripts/benchmark_encoding.py --model ViT-L-14-336
# If >10k img/min: deploy MetaCLIP-L
# If <10k img/min: keep EVA-02-B
```

### Phase 2: Fine-Tuning (Month 2-3)
- Collect 10K labeled images from your cameras
- Fine-tune best model (EVA-02-B or MetaCLIP-L)
- Expect +20-40% accuracy on your queries

### Phase 3: VLM for Hard Queries (Month 4-6)
- Deploy DeepSeek-VL-7B for action verification
- CLIP for fast filter → VLM for hard queries
- Target: smoking, fighting, unusual activities

---

## References

- **MetaCLIP**: https://github.com/facebookresearch/MetaCLIP
- **DFN-CLIP**: https://arxiv.org/abs/2405.17721
- **LLaVA**: https://llava-vl.github.io/
- **DeepSeek-VL**: https://github.com/deepseek-ai/DeepSeek-VL
- **OpenCLIP Model Zoo**: https://github.com/mlfoundations/open_clip

---

**Last Updated:** 2026-05-13  
**Contact:** chinghokuk@gmail.com
