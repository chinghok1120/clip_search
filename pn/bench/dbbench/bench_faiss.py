import os, json, time, resource, gc
import numpy as np, faiss
OUT = os.path.expanduser("~/clip_search/dbbench")
m = json.load(open(f"{OUT}/meta.json"))
N, D, TOPK = m["N"], m["D"], m["TOPK"]; lo, hi = m["sub_lo"], m["sub_hi"]
vecs = np.load(f"{OUT}/vecs.npy"); q = np.load(f"{OUT}/queries.npy"); gt = np.load(f"{OUT}/gt.npy")

t0 = time.time(); index = faiss.IndexFlatIP(D); index.add(vecs); ingest = time.time() - t0
p = f"{OUT}/faiss.index"; faiss.write_index(index, p); disk = os.path.getsize(p)
del vecs; gc.collect()

SP = getattr(faiss, "SearchParametersFlat", faiss.SearchParameters)
params = SP(sel=faiss.IDSelectorRange(lo, hi))
index.search(q[:1], TOPK, params=params)  # warmup
lat = []; ret = np.empty((len(q), TOPK), np.int64)
for i in range(len(q)):
    t1 = time.time(); _, I = index.search(q[i:i+1], TOPK, params=params)
    lat.append((time.time() - t1) * 1000); ret[i] = I[0]
lat.sort()
rec = np.mean([len(set(ret[i]) & set(gt[i])) / TOPK for i in range(len(q))])
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"FAISS flat+IDRange | ingest {ingest:.1f}s | disk {disk/1e6:.0f}MB | RSS {rss:.0f}MB | "
      f"filt-query median {lat[len(lat)//2]:.1f}ms (p95 {lat[int(len(lat)*0.95)]:.1f}) | recall@{TOPK} {rec:.3f}")
