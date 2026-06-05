import gc, numpy as np, torch, torch.nn as nn
from transformers import SiglipVisionModel

name = "google/siglip2-large-patch16-384"
m = SiglipVisionModel.from_pretrained(name, torch_dtype=torch.float32,
        attn_implementation="eager").eval()
res = m.config.image_size
print("image_size", res, flush=True)

class Vis(nn.Module):
    def __init__(self, mm): super().__init__(); self.m = mm
    def forward(self, x):
        f = self.m(x).pooler_output
        return f / f.norm(dim=-1, keepdim=True)

w = Vis(m).eval()
x = torch.randn(1,3,res,res, dtype=torch.float32)
with torch.no_grad(): ref = w(x).numpy()
np.save("val_input384.npy", x.numpy()); np.save("val_ref384.npy", ref)
print("embed", ref.shape[-1], "-> exporting ONNX", flush=True)
torch.onnx.export(w, x, "siglip2_l_384_hf.onnx",
    input_names=["images"], output_names=["embeddings"],
    dynamic_axes={"images":{0:"batch"},"embeddings":{0:"batch"}},
    opset_version=17, do_constant_folding=False)
del w, m; gc.collect()
import onnx; onnx.checker.check_model("siglip2_l_384_hf.onnx")
print("ONNX OK", flush=True)
