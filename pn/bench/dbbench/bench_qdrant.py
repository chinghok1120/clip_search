import os, json, time, subprocess, signal, socket, gc
import numpy as np
from qdrant_client import QdrantClient, models
OUT = os.path.expanduser("~/clip_search/dbbench")
m = json.load(open(f"{OUT}/meta.json"))
N, D, TOPK = m["N"], m["D"], m["TOPK"]; w_lo, w_hi = m["w_lo"], m["w_hi"]
vecs = np.load(f"{OUT}/vecs.npy", mmap_mode="r"); ts = np.load(f"{OUT}/ts.npy"); cam = np.load(f"{OUT}/cam.npy")
q = np.load(f"{OUT}/queries.npy"); gt = np.load(f"{OUT}/gt.npy")

storage = f"{OUT}/qdrant_storage"; os.system(f"rm -rf {storage}")
binp = os.path.expanduser("~/clip_search/qdrant_bin/qdrant")
env = dict(os.environ, QDRANT__STORAGE__STORAGE_PATH=storage, QDRANT__SERVICE__GRPC_PORT="6334",
           QDRANT__SERVICE__HTTP_PORT="6333", QDRANT__TELEMETRY_DISABLED="true")
proc = subprocess.Popen([binp], env=env, cwd=OUT, stdout=open(f"{OUT}/qdrant.log","w"), stderr=subprocess.STDOUT)
def wp(p, t=60):
    s = time.time()
    while time.time()-s < t:
        try: socket.create_connection(("127.0.0.1", p), 1).close(); return True
        except OSError: time.sleep(0.5)
assert wp(6334); time.sleep(2)
cli = QdrantClient(host="127.0.0.1", grpc_port=6334, prefer_grpc=True, timeout=600)
if cli.collection_exists("v"): cli.delete_collection("v")
# m=0 => no HNSW graph => exact filtered search (correct mode for selective time-window filters)
cli.create_collection("v", vectors_config=models.VectorParams(size=D, distance=models.Distance.DOT),
                      hnsw_config=models.HnswConfigDiff(m=0))

t0 = time.time()
for s in range(0, N, 10000):
    e = min(s+10000, N)
    cli.upsert("v", points=models.Batch(ids=list(range(s, e)), vectors=vecs[s:e].tolist(),
        payloads=[{"ts": int(ts[i]), "cam": int(cam[i])} for i in range(s, e)]), wait=(e >= N))
ingest = time.time() - t0
del vecs; gc.collect()
cli.create_payload_index("v", "ts", models.PayloadSchemaType.INTEGER); time.sleep(2)

flt = models.Filter(must=[models.FieldCondition(key="ts", range=models.Range(gte=w_lo, lt=w_hi))])
cli.query_points("v", query=q[0].tolist(), query_filter=flt, limit=TOPK)
lat = []; ret = np.full((len(q), TOPK), -1, np.int64)
for i in range(len(q)):
    t1 = time.time()
    r = cli.query_points("v", query=q[i].tolist(), query_filter=flt, limit=TOPK).points
    lat.append((time.time()-t1)*1000)
    for j, pt in enumerate(r): ret[i, j] = pt.id
lat.sort()
rec = np.mean([len(set(ret[i]) & set(gt[i]))/TOPK for i in range(len(q))])
disk = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(storage) for f in fs)
try: rss = int(open(f"/proc/{proc.pid}/status").read().split("VmRSS:")[1].split()[0]) / 1024
except Exception: rss = -1
print(f"Qdrant exact+filter (m=0) | ingest {ingest:.1f}s | disk {disk/1e6:.0f}MB | RSS {rss:.0f}MB | "
      f"filt-query median {lat[len(lat)//2]:.1f}ms (p95 {lat[int(len(lat)*0.95)]:.1f}) | recall@{TOPK} {rec:.3f}")
proc.send_signal(signal.SIGINT)
try: proc.wait(timeout=30)
except Exception: proc.kill()
