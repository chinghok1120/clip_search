import os, json, time, subprocess, signal, socket
import numpy as np
from qdrant_client import QdrantClient, models
OUT = os.path.expanduser("~/clip_search/dbbench")
m = json.load(open(f"{OUT}/meta.json"))
N, D, TOPK = m["N"], m["D"], m["TOPK"]; w_lo, w_hi = m["w_lo"], m["w_hi"]
lo, hi = m["sub_lo"], m["sub_hi"]
vecs = np.load(f"{OUT}/vecs.npy"); ts = np.load(f"{OUT}/ts.npy")
q = np.load(f"{OUT}/queries.npy"); gt = np.load(f"{OUT}/gt.npy")

storage = f"{OUT}/qdrant_storage"; os.system(f"rm -rf {storage}")
binp = os.path.expanduser("~/clip_search/qdrant_bin/qdrant")
env = dict(os.environ, QDRANT__STORAGE__STORAGE_PATH=storage, QDRANT__SERVICE__GRPC_PORT="6334",
           QDRANT__SERVICE__HTTP_PORT="6333", QDRANT__TELEMETRY_DISABLED="true")
proc = subprocess.Popen([binp], env=env, cwd=OUT, stdout=open(f"{OUT}/qdrant.log","w"), stderr=subprocess.STDOUT)
def wp(p,t=60):
    s=time.time()
    while time.time()-s<t:
        try: socket.create_connection(("127.0.0.1",p),1).close(); return True
        except OSError: time.sleep(0.5)
wp(6334); time.sleep(2)
cli = QdrantClient(host="127.0.0.1", grpc_port=6334, prefer_grpc=True)
if cli.collection_exists("v"): cli.delete_collection("v")
cli.create_collection("v", vectors_config=models.VectorParams(size=D, distance=models.Distance.DOT))
for s in range(0, N, 10000):
    e = min(s+10000, N)
    cli.upsert("v", points=models.Batch(ids=list(range(s,e)), vectors=vecs[s:e].tolist(),
        payloads=[{"ts": int(ts[i])} for i in range(s,e)]), wait=(e>=N))
cli.create_payload_index("v", "ts", models.PayloadSchemaType.INTEGER)
time.sleep(3)

flt = models.Filter(must=[models.FieldCondition(key="ts", range=models.Range(gte=w_lo, lt=w_hi))])
def recall_and_inwin(pts_list):
    rec=[]; inwin=0; tot=0
    for i,pts in enumerate(pts_list):
        ids=[p.id for p in pts]; tot+=len(ids)
        inwin+=sum(1 for x in ids if lo<=x<hi)
        rec.append(len(set(ids)&set(gt[i]))/TOPK)
    return np.mean(rec), inwin/max(tot,1)

# A: filter + approximate (HNSW)
A=[cli.query_points("v", query=q[i].tolist(), query_filter=flt, limit=TOPK).points for i in range(len(q))]
# B: filter + exact
sp=models.SearchParams(exact=True)
B=[cli.query_points("v", query=q[i].tolist(), query_filter=flt, search_params=sp, limit=TOPK).points for i in range(len(q))]
ra,ia=recall_and_inwin(A); rb,ib=recall_and_inwin(B)
print(f"A approx+filter: recall {ra:.3f}, frac-in-window {ia:.3f}")
print(f"B exact +filter: recall {rb:.3f}, frac-in-window {ib:.3f}")
print("sample A ids[0]:", [p.id for p in A[0]][:8])
print("gt[0]:", list(gt[0][:8]), "window id-range:", lo, hi)
proc.send_signal(signal.SIGINT); proc.wait(timeout=30)
