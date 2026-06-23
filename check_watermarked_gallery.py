import math
import torch
import torch.nn.functional as F


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


KEY_PATH = r"weights\fei_r50_protected\watermark_key_theta090.pt"
GALLERY_PATH = r"weights\fei_r50_protected\fei_gallery_wm_theta090.pt"
LABEL = "2"


key = torch_load(KEY_PATH)
gallery = torch_load(GALLERY_PATH)

labels = [str(x) for x in key["labels"]]
idx = labels.index(str(LABEL))

proto = F.normalize(gallery["prototypes"].float(), p=2, dim=1)[idx]
g_clean = F.normalize(key["clean_embeddings"].float(), p=2, dim=1)[idx]
g_wm_key = F.normalize(key["watermarked_embeddings"].float(), p=2, dim=1)[idx]
wi = F.normalize(key["wm_dirs"].float(), p=2, dim=1)[idx]

theta = float(key["theta"])
sin_theta = float(key["sin_theta"])
cos_theta = float(key["cos_theta"]) if "cos_theta" in key else math.cos(theta)

print("========== Watermarked Gallery Check ==========")
print("label:", LABEL)
print("label index:", idx)
print("gallery path:", GALLERY_PATH)
print("key path:", KEY_PATH)
print("-----------------------------------------------")
print("theta:", theta)
print("expected cos(theta):", cos_theta)
print("expected sin(theta):", sin_theta)
print("-----------------------------------------------")
print("cos(gallery_proto, key_watermarked_embedding):", torch.sum(proto * g_wm_key).item())
print("max abs diff between gallery proto and key wm:", torch.max(torch.abs(proto - g_wm_key)).item())
print("cos(clean_embedding, gallery_proto):", torch.sum(g_clean * proto).item())
print("watermark response <gallery_proto, w_i>:", torch.sum(proto * wi).item())