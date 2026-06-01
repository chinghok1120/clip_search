# Jetson Orin Nano Benchmark — EVA-02 & SigLIP2 on the PN

**Date:** 2026-05-29 (PyTorch baselines) · **Updated 2026-05-31** (TensorRT FP16/INT8, resolution/model sweep, SigLIP2 sweep, SigLIP2-on-TRT)
**Hardware:** NVIDIA Jetson Orin Nano 16GB (Super) — host `PNServer` (the PN)
**Goal:** Find a CLIP/SigLIP model + config the PN can run at **≥960 img/min** with acceptable surveillance-search accuracy. (Note: 16:9 CCTV frames are tiled into 2–3 square crops per frame, so the *effective* sustained target is 1,920–2,880 encodes/min — see §13.)

**Bottom line:**
- EVA-02-L/14-**336** cannot reach 960 img/min (786 max, FP16 TRT; INT8 is a dead end — §7). Cleared by **EVA-02-L @ 224px** (1,389) or **EVA-02-B/16** (5,837) — §8.
- **SigLIP2** (open_clip, eager fp16) — §10: **B-16-256 = 6,148** and **L-16-256 = 2,063** clear 960; L-384 (789) and SO400M-378 (461) do not.
- **TensorRT for SigLIP2** — §11–12: the open_clip→trtexec path is **broken** (cos 0.71, an SDPA pooling-head bug — precision-independent). The **working path** is HuggingFace `SiglipVisionModel` + **torch2trt** with **eager attention**: **SigLIP2-L-256 = 3,145 img/min at cos 0.999974** — the strongest accuracy-clearing config found, and the only one that clears even the **3-tile** target (2,880).

---

## 1. Environment

| Component | Value |
|-----------|-------|
| Board | Jetson Orin Nano 16GB (Super), `aarch64` |
| L4T / JetPack | R36.4.0 (JetPack 6.x) |
| CUDA | 12.6 |
| PyTorch | 2.3.0 (NVIDIA Jetson build, CUDA-enabled) |
| open_clip | 3.3.0 |
| Python | 3.10.12 |
| venv | `~/clip_search/venv` created with `--system-site-packages` to inherit the CUDA-enabled system torch |

**Model:** `EVA02-L-14-336` / pretrained `merged2b_s6b_b61k`
- Precision: **fp16**
- Input: 336×336, embed_dim 768
- Model load time: ~10–17 s (cold)

**Method:** `scripts/bench_jetson.py`. Synthetic input tensors of the correct shape (isolates pure encode compute — no JPEG decode / disk I/O). Per batch size: 5 warm-up + 20 timed iterations, **median** reported. Text latency = single query, 20 iterations.

---

## 2. Targets (from CLAUDE.md / PROJECT_PLAN)

| Metric | Target |
|--------|--------|
| Image-encode throughput | **960 img/min** (32 cams × 30 thumbs/min) |
| Per-image (batch) | ~50 ms |
| Text encode | < 50 ms |
| GPU memory | < 8 GB |

---

## 3. Results — three configurations

Three runs, each removing one bottleneck. All fp16, GPU dedicated unless noted.

### 3a. 25W power mode, **with face-detect/recognition dual model contending** (GPU ~440 MHz)

| Batch | img/s | ms/img | img/min | Peak MB |
|-------|-------|--------|---------|---------|
| 1 | 2.7 | 373.3 | 161 | 904 |
| 8 | 2.8 | 351.2 | 171 | 1020 |
| 16 | 2.9 | 339.1 | 177 | 1155 |
| 32 | 3.0 | 329.9 | 182 | 1424 |

Text encode: median **21.9 ms** (p95 26.6). Peak GPU mem: 1424 MB.

### 3b. 25W power mode, **GPU idle** (face models stopped; GPU ~440 MHz)

| Batch | img/s | ms/img | img/min | Peak MB |
|-------|-------|--------|---------|---------|
| 1 | 4.7 | 211.2 | 284 | 904 |
| 8 | 5.2 | 191.9 | 313 | 1020 |
| 16 | 5.3 | 187.6 | 320 | 1155 |
| 32 | 5.3 | 188.7 | 318 | 1424 |

Text encode: median **18.8 ms** (p95 20.2). Peak GPU mem: 1424 MB.

### 3c. **MAXN power mode + `jetson_clocks`** (GPU 918 MHz) ← best

| Batch | img/s | ms/img | img/min | Peak MB |
|-------|-------|--------|---------|---------|
| 1 | 9.9 | 100.7 | 596 | 904 |
| **8** | **10.3** | **97.0** | **618** | 1020 |
| 16 | 9.9 | 101.4 | 592 | 1155 |
| 32 | 9.6 | 104.5 | 574 | 1424 |

Text encode: median **13.8 ms** (p95 14.3). Peak GPU mem: 1424 MB.

---

## 4. Progression summary

| Configuration | GPU clock | img/min (best) | Text ms |
|---|---|---|---|
| 25W + face models contending | ~440 MHz | 182 | 21.9 |
| 25W, GPU idle | ~440 MHz | 320 | 18.8 |
| **MAXN + jetson_clocks** | **918 MHz** | **618** | **13.8** |

Power draw: idle 5.5 W (25W) → 8.1 W (MAXN); under load 12.8 W. The board was never power-limited — the 25W mode capped *clock frequency*, not wattage.

---

## 5. Verdict vs targets — EVA-02-L/14-336, PyTorch eager (best config = MAXN)

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
| Image throughput | 960 img/min | **~600 img/min** | ⚠️ 1.6× short (PyTorch eager) |
| Text encode | < 50 ms | 13.8 ms | ✅ |
| GPU memory | < 8 GB | 1.4 GB | ✅ (huge headroom) |

This is the *eager-mode* result for the 336px model. TensorRT FP16 lifts it to 786 but no further (§7), so the throughput target is ultimately met by changing resolution/model (§8), not by optimizing the 336px engine.

---

## 6. Key findings

1. **GPU-compute-bound.** GPU clock 440→918 MHz (**2.08×**) yielded throughput 320→596 img/min (**1.86×**) — near-linear. The bottleneck is GPU compute, not memory bandwidth. This is the favorable case for TensorRT, which attacks compute directly.

2. **Batching does not help.** Throughput is flat (~10 img/s) from batch 1 to 32 — another signature of being compute-bound rather than launch/throughput-bound. Implication: small batches are fine on the PN; no need to buffer large batches for efficiency.

3. **Power mode matters enormously.** The single biggest lever was MAXN + `jetson_clocks` (2× clock). The default 25W mode silently halves the GPU clock. **The PN must run MAXN + jetson_clocks in production.** (Note: `jetson_clocks` does not persist across reboot — needs a startup service.)

4. **Contention is costly.** The co-resident face-detect/recognition model cost ~45% of throughput (320→177). The PN should not share the GPU with other heavy inference if CLIP throughput matters.

5. **Memory and text latency are non-issues.** 1.4 GB peak (of 16 GB) and 13.8 ms text encode leave large margins — even much larger models fit in memory (see `JETSON_BIGG_FEASIBILITY`), though they won't meet throughput.

---

## 7. TensorRT on EVA-02-L/14-336 — FP16 works, INT8 is a dead end

Pipeline: FP32 PyTorch → **FP32 ONNX** (portable artifact; let TRT pick per-layer precision at build) → TRT engine. Visual tower only (L2-normalized embeddings); validated vs PyTorch by cosine on a held-out image. All at MAXN + jetson_clocks.

| Engine | batch 1 | batch 8 | batch 16 | cos_sim vs PyTorch | Engine size |
|--------|--------:|--------:|---------:|--------------------|------------:|
| **FP16** | **786** | 760 | 682 | 0.999996 ✅ | 625 MB |
| INT8 (calibrator / *implicit*) | 780 | 759 | 683 | 0.999996 | 625 MB |
| INT8 (QDQ / *explicit*) | 635 | **883** | 678 | **0.616 ✗** | 324 MB |

img/min, median of 20 iters. FP16 is the only viable 336px engine: **786 img/min, full accuracy** — still 1.22× short of 960.

**Why INT8 fails here (both routes tried, both rejected):**

1. **Calibrator path (`IInt8EntropyCalibrator2`) did nothing.** Identical 625 MB size, identical throughput, *identical* accuracy to FP16 — TRT kept every layer in FP16. TRT 10.x **deprecated** implicit/calibrator INT8 in favor of explicit quantization; with FP16 fallback allowed, the tactic selector chose FP16 everywhere. The calibration cache was built and ignored.

2. **Explicit QDQ path (onnxruntime static quant) engaged INT8 but did not help.** Inserting QuantizeLinear/DequantizeLinear nodes (restricted to `MatMul`/`Conv`, symmetric INT8 activations, per-channel weights) forced real INT8 — proven by the half-size 324 MB engine and the accuracy drop. But:
   - **No throughput win:** peak 883 img/min (batch 8), batch 1 *slower* (635). EVA-02-L is dominated by non-GEMM ops (rope, LayerNorm, attention softmax) that stay FP16, so the graph constantly reformats INT8↔FP16 around each quantized MatMul; the conversion overhead eats the INT8 compute savings on the Orin Nano. Still under 960.
   - **Accuracy collapsed** (cos 0.616). ViT activations have heavy outliers; per-tensor MinMax calibration sets scales off the outliers and crushes normal-value resolution. Better calibration (Entropy/percentile, SmoothQuant, more images) could recover accuracy but **cannot lift the 883 ceiling** — so it's moot.

**Conclusion: do not pursue INT8 for EVA-02-L on the Orin Nano.** The architecture isn't INT8-friendly on this hardware. The throughput target is met by resolution/model choice instead (§8).

> Gotchas for anyone re-running: TRT 10 rejects a `DequantizeLinear` on a rank-1 bias/LayerNorm tensor — restrict ORT quantization to `MatMul`/`Conv` and set `QuantizeBias: False`. Use `QuantFormat.QDQ` + symmetric activations (TRT requires symmetric). Build with the INT8 flag and **no** calibrator (QDQ scales are explicit).

---

## 8. Resolution / model sweep — what actually clears 960 (PyTorch fp16, MAXN)

Dropping input resolution cuts patch-token count quadratically — the lever INT8 couldn't provide.

| Model @ res | batch 1 | batch 8 (best) | vs 960 | embed dim | Peak MB | Relative accuracy |
|---|--:|--:|:--:|:--:|--:|---|
| EVA-02-L @ 336 (FP16 TRT) | 786 | 760 | ✗ 0.82× | 768 | ~1,600 | **best** |
| **EVA-02-L @ 224** | 1,014 | **1,389** | **✅ 1.45×** | 768 | 942 | **high** (full L weights, fewer tokens) |
| EVA-02-B/16 @ 224 | 1,950 | **5,837** | ✅ 6.1× | 512 | 465 | medium (smaller base model) |

Both clear the target *in eager PyTorch* (no TRT needed); TRT FP16 would add ~1.3× headroom to either.

- **Recommended: EVA-02-L @ 224px** (`EVA02-L-14` / `merged2b_s4b_b131k`). Keeps the full Large weights and 768-dim embeddings — the strongest accuracy that still clears 960 — at 1,389 img/min (1.45× over) and <1 GB. The accuracy-vs-throughput sweet spot for the PN.
- **Max-headroom alternative: EVA-02-B/16** (`EVA02-B-16` / `merged2b_s8b_b131k`). 6× over target, 465 MB, leaves the GPU largely free (e.g. to co-run the face models). Choose it only if L@224's search quality is more than needed.
- **Decision input:** throughput is no longer the constraint — pick between L@224 and B/16 by **search quality**, comparing them in the desktop web tool on representative surveillance queries.

> Note: B/16 and L@224 *batch well* (batch 1→8 roughly doubles/triples throughput), unlike L@336 which is flat/compute-bound. At these smaller sizes batch 1 underutilizes the GPU — run batch ≥8 on the PN.

---

## 9. Reproduce

> **Cleanup note (2026-06-01):** the EVA/INT8 experiment scripts and intermediate artifacts (`export_eva_onnx.py`, `validate_onnx.py`, `quantize_qdq.py`, `build_qdq.py`, `build_int8.py`, all `eva02_l_*` / `siglip_l_visual*` onnx+engines, calibration data) were **removed from the PN** to reclaim ~12 GB. Only the SigLIP2-L deploy set remains: `t2t_siglip2.py` (256), `export_hf_siglip2_384.py` + `siglip2_l_384_hf_fp16.engine` (384), `bench_trt.py`, `bench_jetson.py`. The EVA/INT8 commands below are kept for the historical record but their scripts must be re-created from this doc's snippets if re-run.

```bash
# On the Jetson (PN):
ssh superrx@<PN-ip>
cd ~/clip_search && source venv/bin/activate

# Ensure max performance (requires sudo; not persistent across reboot):
sudo nvpmodel -m 0      # MAXN
sudo jetson_clocks      # pin clocks to max

# PyTorch baselines (bench_jetson.py — synthetic input, isolates encode compute):
python bench_jetson.py --model EVA02-L-14-336 --pretrained merged2b_s6b_b61k --precision fp16  # 336px: ~618
python bench_jetson.py --model EVA02-L-14     --pretrained merged2b_s4b_b131k --precision fp16  # 224px: ~1389 ✅
python bench_jetson.py --model EVA02-B-16      --pretrained merged2b_s8b_b131k --precision fp16  # B/16:  ~5837 ✅

# TensorRT (336px) — FP16 engine, validate + benchmark:
python export_eva_onnx.py                       # FP32 ONNX (visual tower) + validation refs
python validate_onnx.py                         # cosine vs PyTorch (separate process)
trtexec --onnx=eva02_l_visual.onnx --saveEngine=eva02_l_visual_fp16.engine --fp16 \
        --minShapes=images:1x3x336x336 --optShapes=images:8x3x336x336 --maxShapes=images:16x3x336x336
python bench_trt.py --engine eva02_l_visual_fp16.engine   # ~786, cos 0.999996

# INT8 QDQ (documented dead end — engages INT8 but no speedup + accuracy loss):
python quantize_qdq.py --limit 64               # FP32 ONNX -> QDQ ONNX (MatMul/Conv only)
python build_qdq.py                             # explicit-INT8 engine (no calibrator)
python bench_trt.py --engine eva02_l_visual_qdq.engine    # peak 883, cos 0.616 ✗
```

Scripts (on the PN at `~/clip_search/`): `bench_jetson.py`, `export_eva_onnx.py`, `validate_onnx.py`, `bench_trt.py`, `quantize_qdq.py`, `build_qdq.py`. `bench_jetson.py` is also in `scripts/`.

---

## 10. SigLIP2 sweep (open_clip, PyTorch fp16, MAXN)

Same method as §3 (`bench_jetson.py`, synthetic input, median of 20). Text encode skipped (SigLIP needs `transformers` for its tokenizer — not installed at sweep time; image encoding doesn't need it).

| Model @ res | batch 1 | batch 8 (best) | vs 960 | embed dim | Peak MB | Notes |
|---|--:|--:|:--:|:--:|--:|---|
| **ViT-B-16-SigLIP2 @ 256** | 4,547 | **6,148** | ✅ 6.4× | 768 | 864 | max headroom; richer embed than EVA-B (512) |
| **ViT-L-16-SigLIP2 @ 256** | **2,063** | 1,962 | ✅ 2.1× | 1024 | 1,879 | strongest accuracy that clears 960 (eager) |
| ViT-L-16-SigLIP2 @ 384 | 789 | 716 | ✗ 0.82× | 1024 | 2,014 | borderline — same wall as EVA-L@336 |
| ViT-SO400M-14-SigLIP2 @ 378 | 461 | 413 | ✗ 0.48× | 1152 | 2,630 | desktop accuracy champ — too heavy for PN |

- **Batching:** B-256 batches well (4.5k→6.1k); L+ do **not** (flat/declining — compute-bound, batch-1 is best). Run batch 1–8.
- **embed_dim** is higher than EVA across the board (768/1024/1152 vs EVA-L 768 / EVA-B 512) → richer embeddings.
- SO400M/L-384 sit in the "max accuracy, fails throughput" bucket — like EVA-L@336.

---

## 11. TensorRT for SigLIP2 — the open_clip path is broken, the HF path works

We wanted TRT speed on **SigLIP2-L-256** (best accuracy that clears 960 in eager) to gain headroom — specifically to clear the 16:9 **3-tile** target (2,880/min, §13), which eager (2,063) does not.

### 11a. open_clip → trtexec — DEAD END (precision-independent accuracy collapse)

Same export/validate harness as EVA (§7): FP32 ONNX → trtexec → cosine vs PyTorch.

| Engine (open_clip ViT-L-16-SigLIP2-256) | batch 1 | batch 8 | cos_sim vs PyTorch | Verdict |
|---|--:|--:|:--:|---|
| trtexec **FP16** | 2,953 | 2,614 | **0.721 ✗** | degraded |
| trtexec **BF16** | 2,602 | 2,587 | **0.706 ✗** | degraded |
| trtexec **FP32** | — | — | **0.717 ✗** | degraded |
| ONNX-Runtime FP32 (sanity) | — | — | 1.000 (4e-7 abs) ✓ | export is faithful |
| PyTorch FP16 (sanity) | 2,063 | 1,962 | 0.999999 ✓ | model is fine in fp16 |

Speed was great (~2,600), but **every TRT precision degraded identically** while the ONNX and PyTorch-fp16 references were perfect → **not a precision problem, a TRT op-compilation bug.** open_clip wraps SigLIP as a **timm** model whose attention-pooling head (`AttentionPoolLatent`) uses `F.scaled_dot_product_attention`; TRT mis-compiles that SDPA subgraph (cf. [pytorch/TensorRT#3823](https://github.com/pytorch/TensorRT/issues/3823), non-contiguous SDPA output). Disabling torch's MHA fast-path had no effect — confirming the pooler is timm-SDPA, not `nn.MultiheadAttention`.

### 11b. HF `SiglipVisionModel` + torch2trt — WORKS (full accuracy, fastest yet)

HuggingFace's SigLIP implementation uses a **different** graph (MAP head = `nn.MultiheadAttention`), and `torch2trt`'s ONNX path converts it cleanly — *if* attention is forced to `eager`.

| Path (batch 1) | img/min | cos_sim vs PyTorch | Peak MB |
|---|--:|:--:|--:|
| HF `google/siglip-large-patch16-256` (v1) → torch2trt FP16 | 2,803 | 0.999995 ✓ | — |
| **HF `google/siglip2-large-patch16-256` → torch2trt FP16 (eager attn)** | **3,145** | **0.999974 ✓** | 2,306 |
| HF `google/siglip2-large-patch16-384` → export-ONNX → **trtexec** FP16 (eager) | 1,151 | 0.999917 ✓ | — |

**3,145 img/min at cos 0.999974** — 1.52× over open_clip eager fp16 (2,063), and it clears the 3-tile target.

**On the 384 variant:** clears only the 1-tile target (1,151 vs 960). The real trade is *one* 384px view/frame (1,151/min) vs *three* 256px tiles/frame (3,145/min); for 1920×1080 frames with small subjects, 3×256 tiles preserve more pixels-on-target, so **256 stays the pick**. Build note: **torch2trt OOMs building the 384 graph on 16 GB** (holds torch + 1.26 GB ONNX + TRT builder at once → RC=137/SIGKILL even at 1 GB workspace). Memory-safe path for larger graphs: **split it** — export FP32 ONNX in one process (torch frees on exit) → build with `trtexec` separately. This also proved trtexec compiles the HF-**eager** ONNX correctly (cos 0.999917) — a second working TRT path besides torch2trt. These are the same Google SigLIP2 weights open_clip's `webli` ports, so search quality should match what was benchmarked in the desktop tool (re-index with the HF model).

**Working recipe:**
```python
from transformers import SiglipVisionModel   # v1 class; fixed-res SigLIP2 is config type siglip_vision_model
import torch2trt, torch
m = SiglipVisionModel.from_pretrained("google/siglip2-large-patch16-256",
        torch_dtype=torch.float32, attn_implementation="eager").eval().cuda()   # eager is mandatory
x = torch.randn(1,3,256,256, device="cuda")
trt = torch2trt.torch2trt(m, [x], fp16_mode=True, use_onnx=True, max_workspace_size=(1024**3)*3)
# trt(x).pooler_output  -> the image embedding
```
Script on PN: `~/clip_search/t2t_siglip2.py`. Deps added to the PN venv: `transformers==4.51.3`, `torch2trt` (NVIDIA-AI-IOT), `onnx_graphsurgeon`, all with `numpy<2` pinned.

---

## 12. Blockers we hit → root cause → solution

Everything that broke during the EVA + SigLIP + TRT investigation, why, and the fix.

| # | Blocker (symptom) | Root cause (the reason) | Solution |
|---|---|---|---|
| 1 | Throughput stuck ~320/min, "power-limited?" | Default **25W** nvpmodel silently **caps GPU clock** at 440 MHz (not wattage) | `sudo nvpmodel -m 0` (MAXN) + `sudo jetson_clocks` → 918 MHz, ~1.9× throughput. (MAXN persists reboot; jetson_clocks does not — needs a startup unit) |
| 2 | EVA-L@336 INT8 **calibrator** gave identical size/speed/accuracy to FP16 | TRT 10.x **deprecated** implicit/calibrator INT8; with FP16 fallback, tactic selector kept every layer FP16 | Abandon calibrator path; use explicit QDQ instead (then §7 ruled INT8 out entirely) |
| 3 | EVA-L@336 INT8 **QDQ** → cos 0.616 **and** no speedup | ViT activation **outliers** wreck per-tensor MinMax scales; non-GEMM ops (rope/LN/softmax) stay FP16 so graph **reformats INT8↔FP16** around each MatMul, eating the win | Abandon INT8 on ViT for this board; get headroom via **resolution reduction** instead (§8) |
| 4 | QDQ build rejected: `DequantizeLinear` on a rank-1 bias | ORT had quantized **LayerNorm biases**; TRT won't dequantize rank-1 tensors | Restrict `op_types_to_quantize=["MatMul","Conv"]`, set `QuantizeBias=False` |
| 5 | EVA-L@336 maxes at 786 even on FP16 TRT | Model is **compute-bound** at 336px (576 patch tokens); TRT FP16 only ~1.3× | Drop to **224px** → fewer tokens → EVA-L@224 = 1,389 ✅ |
| 6 | open_clip SigLIP2-L → TRT: cos **0.71** in FP16/BF16/FP32 | **timm** `AttentionPoolLatent` head uses `F.scaled_dot_product_attention`; TRT mis-compiles that SDPA subgraph (precision-independent op bug) | Don't export open_clip SigLIP via trtexec; use the **HF model + torch2trt** path (§11b) |
| 7 | Disabling torch MHA fast-path didn't fix #6 | The pooler isn't `nn.MultiheadAttention` at all — it's timm SDPA | Confirmed a different export path was needed (→ HF) |
| 8 | `transformers` 5.9 prints "PyTorch was not found" (`is_torch_available()=False`) | transformers **5.x requires torch > 2.3**; with system torch 2.3.0 it disables the torch backend → torch models won't load | Pin **`transformers==4.51.3`** (any 4.x detects torch 2.3) |
| 9 | torch2trt `use_onnx`: `ModuleNotFoundError: onnx_graphsurgeon` | torch2trt's ONNX path needs graphsurgeon (not a hard dep) | `pip install onnx-graphsurgeon` |
| 10 | `Siglip2VisionModel` load → state_dict size mismatch (`[1024,3,16,16]` Conv vs `[1024,768]` Linear) | `Siglip2VisionModel` is the **NaFlex** variant (Linear patch embed); the **fixed-res** SigLIP2 checkpoint has a Conv2d patch embed and config type `siglip_vision_model` | Load fixed-res SigLIP2 with the **v1 `SiglipVisionModel`** class |
| 11 | torch2trt: `TypeError ... translating scaled_dot_product_attention` | transformers 4.51 defaults `attn_implementation="sdpa"`; torch2trt's SDPA converter chokes | Pass **`attn_implementation="eager"`** (mathematically identical; traces to matmul+softmax) |
| 12 | Risk: installing transformers could pull **numpy 2** and break torch 2.3 | torch 2.3.0 needs numpy < 2; pip would otherwise upgrade it | Add **`"numpy<2"`** to every `pip install` (held at 1.26.4 throughout) |
| 13 | `dusty-nv/clip_trt` silently fell back to Transformers (no TRT) | Its `CLIPModel` **disables TRT below 20 GB RAM**; Orin Nano = 16 GB | Bypass the wrapper — call `torch2trt.torch2trt(...)` directly (what `t2t_siglip2.py` does) |

**Two recurring lessons:** (a) **MAXN+jetson_clocks is mandatory** — half the GPU otherwise. (b) **SDPA is the TRT villain** for these ViTs — both dead ends (open_clip timm pooler, transformers-4.51 encoder default) were `scaled_dot_product_attention`; route around it (eager attention / a graph that doesn't use it).

---

## 13. Consolidated leaderboard — every EVA + SigLIP config benchmarked

All MAXN + jetson_clocks. "best img/min" = best across batch 1/8/16. PyTorch = open_clip eager fp16 unless noted.

| Model @ res | engine | best img/min | accuracy (cos vs fp32) | embed dim | Peak MB | clears 960? |
|---|---|--:|:--:|:--:|--:|:--:|
| ViT-B-16-SigLIP2 @ 256 | PyTorch fp16 | **6,148** | — | 768 | 864 | ✅ 6.4× |
| EVA-02-B/16 @ 224 | PyTorch fp16 | **5,837** | — | 512 | 465 | ✅ 6.1× |
| **SigLIP2-L-16 @ 256** | **HF+torch2trt FP16** | **3,145** | **0.999974 ✓** | 1024 | 2,306 | ✅ 3.3× |
| SigLIP-L-16 @ 256 (v1) | HF+torch2trt FP16 | 2,803 | 0.999995 ✓ | 768 | — | ✅ 2.9× |
| ViT-L-16-SigLIP2 @ 256 | PyTorch fp16 | 2,063 | (0.999999 vs fp32) | 1024 | 1,879 | ✅ 2.1× |
| ViT-L-16-SigLIP2 @ 256 | open_clip→trtexec | ~2,600 | **0.71 ✗** | 1024 | — | (broken) |
| EVA-02-L/14 @ 224 | PyTorch fp16 | 1,389 | — | 768 | 942 | ✅ 1.45× |
| **SigLIP2-L-16 @ 384** | **HF→trtexec FP16** | 1,151 | 0.999917 ✓ | 1024 | — | ✅ 1.2× (1-tile only) |
| ViT-L-16-SigLIP2 @ 384 | PyTorch fp16 | 789 | — | 1024 | 2,014 | ✗ 0.82× |
| EVA-02-L/14 @ 336 | TRT FP16 | 786 | 0.999996 ✓ | 768 | ~1,600 | ✗ 0.82× |
| EVA-02-L/14 @ 336 | PyTorch fp16 | 618 | — | 768 | 1,424 | ✗ 0.64× |
| ViT-SO400M-14-SigLIP2 @ 378 | PyTorch fp16 | 461 | — | 1152 | 2,630 | ✗ 0.48× |

**Against the 16:9 tiling targets** (encode 2–3 crops/frame → 1,920 / 2,880 needed):

| Config | best/min | 1-tile (960) | 2-tile (1,920) | 3-tile (2,880) |
|---|--:|:--:|:--:|:--:|
| SigLIP2-B-256 / EVA-B16 | 5,837–6,148 | ✅ | ✅ | ✅ |
| **SigLIP2-L-256 (HF TRT)** | **3,145** | ✅ | ✅ | ✅ |
| SigLIP2-L-256 (eager) | 2,063 | ✅ | ✅ | ✗ |
| EVA-L@224 (eager) | 1,389 | ✅ | ✗ | ✗ |

**Recommendation:** for the best accuracy that clears all tiling targets, deploy **SigLIP2-L-16-256 via HF + torch2trt (eager, FP16)** — 3,145 img/min, cos 0.999974, 2.3 GB. If GPU headroom / co-running other models matters more than top accuracy, **SigLIP2-B-256** or **EVA-B/16** give ~6× with a smaller footprint.
