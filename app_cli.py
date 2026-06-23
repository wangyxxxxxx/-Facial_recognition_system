import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


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


def load_backbone(network, weights, device):
    from backbones import get_model

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

    print("========== Model Load ==========")
    print("network:", network)
    print("weights:", weights)
    print("missing keys:", len(missing))
    print("unexpected keys:", len(unexpected))

    model.to(device)
    model.eval()
    return model


def load_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    return img


@torch.no_grad()
def extract_embedding(model, image_path, device):
    img = load_image(image_path, device)
    feat = model(img)

    if isinstance(feat, (tuple, list)):
        feat = feat[0]

    feat = F.normalize(feat.float(), p=2, dim=1)
    return feat.squeeze(0).detach().cpu()


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
            raise ValueError(f"无法识别 gallery keys: {gallery.keys()}")
    elif isinstance(gallery, torch.Tensor):
        prototypes = gallery.float()
        labels = list(range(prototypes.shape[0]))
    else:
        raise ValueError(f"不支持的 gallery 类型: {type(gallery)}")

    prototypes = F.normalize(prototypes, p=2, dim=1)
    return labels, prototypes


def find_label_index(labels, target_label):
    target_str = str(target_label)

    for i, lab in enumerate(labels):
        if str(lab) == target_str:
            return i

    raise ValueError(
        f"在 labels 中找不到 label={target_label}。"
        f"前20个 labels: {labels[:20]}"
    )


class ArcFaceCLI:
    def __init__(self, weights, network="r50", device="cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA 不可用，自动切换到 CPU")
            device = "cpu"

        self.device = torch.device(device)
        self.network = network
        self.weights = weights
        self.model = load_backbone(network, weights, self.device)

    def predict(self, image_path, gallery_path, topk=5):
        labels, gallery_emb = load_gallery(gallery_path)
        z = extract_embedding(self.model, image_path, self.device).unsqueeze(0)

        scores = torch.matmul(z, gallery_emb.T).squeeze(0)
        topk = min(topk, scores.numel())

        vals, idxs = torch.topk(scores, k=topk)

        pred_idx = idxs[0].item()
        pred_label = labels[pred_idx]
        top1_cosine = vals[0].item()

        top_results = []
        for v, idx in zip(vals, idxs):
            idx = idx.item()
            top_results.append({
                "label": labels[idx],
                "cosine": v.item()
            })

        return {
            "pred_label": pred_label,
            "top1_cosine": top1_cosine,
            "topk": top_results,
        }

    def detect_watermark(self, image_path, key_path, label, threshold=0.085):
        key = torch_load(key_path, map_location="cpu")

        labels = key["labels"]
        wm_dirs = key["wm_dirs"].float()
        clean_embeddings = key["clean_embeddings"].float()
        watermarked_embeddings = key["watermarked_embeddings"].float()

        theta = float(key["theta"])
        sin_theta = float(key.get("sin_theta", torch.sin(torch.tensor(theta)).item()))

        idx = find_label_index(labels, label)

        wi = F.normalize(wm_dirs[idx], p=2, dim=0)
        g_clean = F.normalize(clean_embeddings[idx], p=2, dim=0)
        g_wm = F.normalize(watermarked_embeddings[idx], p=2, dim=0)

        z = extract_embedding(self.model, image_path, self.device)

        s_wm = torch.sum(z * wi).item()
        cos_to_clean = torch.sum(z * g_clean).item()
        cos_to_wm = torch.sum(z * g_wm).item()

        detected = s_wm > threshold

        return {
            "label": label,
            "label_index": idx,
            "theta": theta,
            "sin_theta": sin_theta,
            "threshold": threshold,
            "s_wm": s_wm,
            "cos_clean": cos_to_clean,
            "cos_wm": cos_to_wm,
            "cos_wm_minus_clean": cos_to_wm - cos_to_clean,
            "detected": detected,
        }


def print_predict_result(result):
    print()
    print("========== Predict Result ==========")
    print("pred_label:", result["pred_label"])
    print("top1_cosine:", f"{result['top1_cosine']:.6f}")
    print()
    print(f"Top-{len(result['topk'])}:")
    for item in result["topk"]:
        print(f"  label={item['label']}  cosine={item['cosine']:.6f}")


def print_watermark_result(result):
    print()
    print("========== Watermark Detection ==========")
    print("target label:", result["label"])
    print("label index:", result["label_index"])
    print("theta:", result["theta"])
    print("expected sin(theta):", result["sin_theta"])
    print("threshold:", result["threshold"])
    print("-----------------------------------------")
    print("watermark score s_wm = <z, w_i>:", result["s_wm"])
    print("cos(z, clean_gallery):", result["cos_clean"])
    print("cos(z, watermarked_gallery):", result["cos_wm"])
    print("cos(z, wm) - cos(z, clean):", result["cos_wm_minus_clean"])
    print("-----------------------------------------")
    print("detected:", int(result["detected"]))
    if result["detected"]:
        print("result: watermark detected")
    else:
        print("result: watermark not detected")


def build_parser():
    parser = argparse.ArgumentParser(
        description="ArcFace 本地 CPU 命令行工具：普通识别 + 水印检测"
    )

    parser.add_argument(
        "mode",
        choices=["predict", "detect", "both"],
        help="predict=普通识别，detect=水印检测，both=同时执行"
    )

    parser.add_argument("--image", required=True, help="输入图片路径")

    parser.add_argument(
        "--weights",
        default=r"weights\fei_r50-train\model.pt",
        help="ArcFace 模型权重路径"
    )

    parser.add_argument(
        "--gallery",
        default=r"weights\fei_r50-train\fei_gallery.pt",
        help="gallery 路径"
    )

    parser.add_argument(
        "--key",
        default=r"weights\fei_r50_protected\watermark_key_theta090.pt",
        help="水印 key 路径"
    )

    parser.add_argument(
        "--label",
        default=None,
        help="水印检测目标 label。detect 模式必须提供；both 模式不提供则默认使用 pred_label"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.085,
        help="水印检测阈值，默认 0.085"
    )

    parser.add_argument("--network", default="r50", help="backbone 类型，默认 r50")
    parser.add_argument("--device", default="cpu", help="cpu 或 cuda，默认 cpu")
    parser.add_argument("--topk", type=int, default=5, help="Top-K 输出数量")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"图片不存在: {args.image}")

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"权重不存在: {args.weights}")

    app = ArcFaceCLI(
        weights=args.weights,
        network=args.network,
        device=args.device,
    )

    pred_result = None

    if args.mode in ["predict", "both"]:
        if not os.path.exists(args.gallery):
            raise FileNotFoundError(f"gallery 不存在: {args.gallery}")

        pred_result = app.predict(
            image_path=args.image,
            gallery_path=args.gallery,
            topk=args.topk,
        )
        print_predict_result(pred_result)

    if args.mode in ["detect", "both"]:
        if not os.path.exists(args.key):
            raise FileNotFoundError(f"watermark key 不存在: {args.key}")

        label = args.label

        if label is None:
            if args.mode == "both" and pred_result is not None:
                label = pred_result["pred_label"]
                print()
                print(f"[Info] 未提供 --label，自动使用预测 label={label} 做水印检测")
            else:
                raise ValueError("detect 模式必须提供 --label，例如 --label 1")

        wm_result = app.detect_watermark(
            image_path=args.image,
            key_path=args.key,
            label=label,
            threshold=args.threshold,
        )
        print_watermark_result(wm_result)


if __name__ == "__main__":
    main()