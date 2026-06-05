import time, torch, numpy as np
from transformers import SiglipVisionModel
import torch2trt

name = "google/siglip2-large-patch16-256"
print("loading", name, flush=True)
m = SiglipVisionModel.from_pretrained(name, torch_dtype=torch.float32, attn_implementation="eager").eval().cuda()
res = m.config.image_size
print("image_size", res, flush=True)
x = torch.randn(1,3,res,res, dtype=torch.float32, device="cuda")
# sanity forward (catch naflex-style extra-arg requirement early)
with torch.no_grad():
    a = m(x).pooler_output
print("torch forward OK, embed", a.shape[-1], flush=True)

torch.cuda.reset_peak_memory_stats()
print("converting via torch2trt...", flush=True)
trt = torch2trt.torch2trt(m, [x], fp16_mode=True, max_workspace_size=(1024**3)*3, use_onnx=True)

with torch.no_grad():
    bo = trt(x)
    b = bo.pooler_output if hasattr(bo,"pooler_output") else (bo["pooler_output"] if isinstance(bo,dict) else bo)
a=a.float(); b=b.float()
an=a/a.norm(dim=-1,keepdim=True); bn=b/b.norm(dim=-1,keepdim=True)
print(f"ACCURACY  cos_sim={float((an*bn).sum(-1).mean()):.6f} max_abs_diff={(a-b).abs().max():.2e}", flush=True)

def prof(model, runs=20):
    with torch.no_grad():
        for _ in range(5): model(x)
        torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(runs): model(x)
        torch.cuda.synchronize()
    return (time.perf_counter()-t)/runs
ms_trt=prof(trt)
print(f"THROUGHPUT batch1  trt={1000*ms_trt:.1f}ms ({60/ms_trt:.0f}/min)", flush=True)
print(f"PEAK GPU mem: {torch.cuda.max_memory_allocated()/1e6:.0f} MB", flush=True)
