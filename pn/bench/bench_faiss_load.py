import os, time
import numpy as np
import faiss

N = 1_382_400          # 32 cams * (86400/2) thumbs/day
D = 1024               # SigLIP2-L embedding dim
PATH = os.path.expanduser("~/clip_search/bench_shard.faiss")
CHUNK = 100_000

print(f"faiss {faiss.__version__}, threads={faiss.omp_get_max_threads()}", flush=True)

# ---- build a realistic 1-day flat shard ----
t0 = time.time()
index = faiss.IndexFlatIP(D)
rng = np.random.default_rng(0)
n = 0
while n < N:
    m = min(CHUNK, N - n)
    x = rng.standard_normal((m, D), dtype=np.float32)
    faiss.normalize_L2(x)
    index.add(x)
    n += m
    del x
print(f"[build] {index.ntotal} vecs x {D}  in {time.time()-t0:.1f}s", flush=True)

# ---- write to NVMe ----
t0 = time.time()
faiss.write_index(index, PATH)
os.sync()
sz = os.path.getsize(PATH)
print(f"[write] {sz/1e9:.2f} GB on disk in {time.time()-t0:.1f}s", flush=True)
del index

# ---- evict this file from page cache (cold read, no root) ----
fd = os.open(PATH, os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)

# ---- COLD load (from disk) ----
t0 = time.time()
index = faiss.read_index(PATH)
cold = time.time() - t0
print(f"[load COLD] {cold:.2f}s  -> {sz/1e9/cold:.2f} GB/s", flush=True)

# ---- single-query search latency (the interactive case) ----
rng = np.random.default_rng(1)
q = rng.standard_normal((32, D), dtype=np.float32); faiss.normalize_L2(q)
index.search(q[:1], 20)   # warmup
lat = []
for i in range(32):
    t0 = time.time(); index.search(q[i:i+1], 20); lat.append((time.time()-t0)*1000)
lat.sort()
print(f"[search nq=1] top20 over {N:,}: median {lat[len(lat)//2]:.1f} ms  (min {lat[0]:.1f}, max {lat[-1]:.1f})", flush=True)

# ---- batched search (shows multithread throughput) ----
t0 = time.time(); index.search(q[:16], 20); b = (time.time()-t0)*1000
print(f"[search nq=16] {b:.1f} ms total = {b/16:.1f} ms/query", flush=True)

# ---- WARM load (now in cache) for comparison ----
del index
t0 = time.time()
index = faiss.read_index(PATH)
warm = time.time() - t0
print(f"[load WARM] {warm:.2f}s  -> {sz/1e9/warm:.2f} GB/s", flush=True)

os.remove(PATH)
print("[cleanup] removed shard", flush=True)
