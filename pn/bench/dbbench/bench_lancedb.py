import os, json, time, resource, gc
import numpy as np, lancedb, pyarrow as pa
OUT = os.path.expanduser("~/clip_search/dbbench")
m = json.load(open(f"{OUT}/meta.json"))
N, D, TOPK = m["N"], m["D"], m["TOPK"]; w_lo, w_hi = m["w_lo"], m["w_hi"]
vecs = np.load(f"{OUT}/vecs.npy", mmap_mode="r")           # mmap: don't pull 4GB resident
ts = np.load(f"{OUT}/ts.npy"); cam = np.load(f"{OUT}/cam.npy")
q = np.load(f"{OUT}/queries.npy"); gt = np.load(f"{OUT}/gt.npy")
ids = np.arange(N, dtype=np.int64)

dbdir = f"{OUT}/lancedb"; os.system(f"rm -rf {dbdir}")
db = lancedb.connect(dbdir)
def chunk(s, e):
    return pa.table({
        "id": pa.array(ids[s:e]),
        "vector": pa.FixedSizeListArray.from_arrays(pa.array(np.ascontiguousarray(vecs[s:e]).reshape(-1)), D),
        "ts": pa.array(ts[s:e]), "cam": pa.array(cam[s:e].astype(np.int32)),
    })
CH = 100_000
t0 = time.time()
tbl = db.create_table("v", data=chunk(0, min(CH, N)))
for s in range(CH, N, CH):
    tbl.add(chunk(s, min(s + CH, N)))
tbl.create_scalar_index("ts")
ingest = time.time() - t0
gc.collect()

where = f"ts >= {w_lo} and ts < {w_hi}"
tbl.search(q[0].tolist()).where(where, prefilter=True).limit(TOPK).select(["id"]).to_list()
lat = []; ret = np.full((len(q), TOPK), -1, np.int64)
for i in range(len(q)):
    t1 = time.time()
    r = tbl.search(q[i].tolist()).where(where, prefilter=True).limit(TOPK).select(["id"]).to_list()
    lat.append((time.time() - t1) * 1000)
    for j, row in enumerate(r): ret[i, j] = row["id"]
lat.sort()
rec = np.mean([len(set(ret[i]) & set(gt[i])) / TOPK for i in range(len(q))])
disk = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(dbdir) for f in fs)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"LanceDB flat+ts-index | ingest {ingest:.1f}s | disk {disk/1e6:.0f}MB | RSS {rss:.0f}MB | "
      f"filt-query median {lat[len(lat)//2]:.1f}ms (p95 {lat[int(len(lat)*0.95)]:.1f}) | recall@{TOPK} {rec:.3f}")
