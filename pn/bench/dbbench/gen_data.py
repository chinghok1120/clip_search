import os, json, time
import numpy as np

N = int(os.environ.get("N", "100000"))
D, NQ, TOPK, SPAN_DAYS = 1024, 50, 20, 30
OUT = os.path.expanduser("~/clip_search/dbbench")
rng = np.random.default_rng(42)
t0 = time.time()

# clustered embeddings (mixture of gaussians on the sphere) -> realistic ANN structure
K = max(64, N // 500)
centers = rng.standard_normal((K, D)).astype(np.float32)
centers /= np.linalg.norm(centers, axis=1, keepdims=True)
assign = rng.integers(0, K, N)
vecs = np.empty((N, D), np.float32)
CH = 100_000
for s in range(0, N, CH):
    e = min(s + CH, N)
    nz = rng.standard_normal((e - s, D)).astype(np.float32)
    nz /= np.linalg.norm(nz, axis=1, keepdims=True)
    v = centers[assign[s:e]] + 0.35 * nz
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    vecs[s:e] = v
del assign

t_start = 1_700_000_000; span = SPAN_DAYS * 86400
ts = np.sort(rng.integers(t_start, t_start + span, size=N)).astype(np.int64)
cam = rng.integers(0, 32, size=N).astype(np.int16)
# queries near random clusters (realistic: query resembles indexed content)
qa = rng.integers(0, K, NQ)
qn = rng.standard_normal((NQ, D)).astype(np.float32); qn /= np.linalg.norm(qn, axis=1, keepdims=True)
queries = centers[qa] + 0.35 * qn; queries /= np.linalg.norm(queries, axis=1, keepdims=True)
queries = queries.astype(np.float32)

w_lo = int(t_start + span // 2); w_hi = int(w_lo + 86400)
sub = np.where((ts >= w_lo) & (ts < w_hi))[0]
sims = queries @ vecs[sub].T
gt = sub[np.argsort(-sims, axis=1)[:, :TOPK]].astype(np.int64)

np.save(f"{OUT}/vecs.npy", vecs); np.save(f"{OUT}/ts.npy", ts); np.save(f"{OUT}/cam.npy", cam)
np.save(f"{OUT}/queries.npy", queries); np.save(f"{OUT}/gt.npy", gt)
json.dump({"N": N, "D": D, "NQ": NQ, "TOPK": TOPK, "w_lo": w_lo, "w_hi": w_hi,
           "sub_lo": int(sub[0]), "sub_hi": int(sub[-1] + 1), "sub_size": int(sub.size)},
          open(f"{OUT}/meta.json", "w"))
print(f"gen N={N} K={K} sub_size={sub.size} id_range[{sub[0]},{sub[-1]}] in {time.time()-t0:.1f}s")
