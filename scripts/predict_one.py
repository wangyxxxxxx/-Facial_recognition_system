import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from backbones import get_model


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v
    return new_state


def load_model(weights, network, device):
    model = get_model(network, fp16=False)

    ckpt = torch_load(weights, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]
        elif "backbone" in ckpt:
            ckpt = ckpt["backbone"]

    ckpt = strip_module_prefix(ckpt)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)

    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))

    model.to(device)
    model.eval()
    return model


def load_image(path, device):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
    ])
    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    return img


@torch.no_grad()
def extract_embedding(model, image_path, device):
    img = load_image(image_path, device)
    feat = model(img)

    if isinstance(feat, (tuple, list)):
        feat = feat[0]

    feat = F.normalize(feat.float(), p=2, dim=1)
    return feat


def load_gallery(path):
    gallery = torch_load(path, map_location="cpu")

    if isinstance(gallery, dict):
        if "prototypes" in gallery:
            prototypes = gallery["prototypes"].float()
            labels = gallery.get("labels", list(range(prototypes.shape[0])))
        elif "features" in gallery:
            prototypes = gallery["features"].float()
            labels = gallery.get("labels", list(range(prototypes.shape[0])))
        elif "embeddings" in gallery:
            prototypes = gallery["embeddings"].float()
            labels = gallery.get("labels", list(range(prototypes.shape[0])))
        else:
            raise ValueError(f"Unknown gallery keys: {gallery.keys()}")
    elif isinstance(gallery, torch.Tensor):
        prototypes = gallery.float()
        labels = list(range(prototypes.shape[0]))
    else:
        raise ValueError(f"Unsupported gallery type: {type(gallery)}")

    prototypes = F.normalize(prototypes, p=2, dim=1)
    return labels, prototypes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--gallery", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--network", default="r50")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    device = torch.device(args.device)

    model = load_model(args.weights, args.network, device)
    labels, gallery_emb = load_gallery(args.gallery)

    z = extract_embedding(model, args.image, device).cpu()

    scores = torch.matmul(z, gallery_emb.T).squeeze(0)
    topk = min(args.topk, scores.numel())

    vals, idxs = torch.topk(scores, k=topk)

    pred_idx = idxs[0].item()
    pred_label = labels[pred_idx]
    top1_cosine = vals[0].item()

    print()
    print("pred_label:", pred_label)
    print("top1_cosine:", f"{top1_cosine:.6f}")
    print()
    print(f"Top-{topk}:")
    for v, idx in zip(vals, idxs):
        idx = idx.item()
        print(f"  label={labels[idx]}  cosine={v.item():.6f}")


if __name__ == "__main__":
    main()
