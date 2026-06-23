import argparse
import math
import os
import sys
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
    """
    适配 insightface/recognition/arcface_torch 常见结构：
        from backbones import get_model
    请把本脚本放在 arcface_torch 根目录下运行。
    """
    try:
        from backbones import get_model
    except Exception as e:
        print("无法 import backbones.get_model。")
        print("请把 detect_embedding_watermark.py 放到 arcface_torch 目录下运行，")
        print("或者确认当前目录能 import: from backbones import get_model")
        raise e

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


def find_label_index(labels, target_label):
    """
    labels 可能是 int，也可能是 str。
    为了兼容，统一转成 str 比较一次。
    """
    target_str = str(target_label)

    for i, lab in enumerate(labels):
        if str(lab) == target_str:
            return i

    raise ValueError(
        f"在 watermark_key 的 labels 中找不到 label={target_label}。"
        f"前20个 labels: {labels[:20]}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="待检测图片，例如反演图 results/direct_type1_2.png")
    parser.add_argument("--weights", type=str, required=True,
                        help="ArcFace backbone 权重，例如 output/fei_r50_protected/model.pt")
    parser.add_argument("--network", type=str, default="r50",
                        help="backbone 名称，例如 r50, r100, mobilefacenet")
    parser.add_argument("--key", type=str, required=True,
                        help="make_embedding_watermark_gallery.py 生成的 watermark_key.pt")
    parser.add_argument("--label", type=str, required=True,
                        help="目标身份 label，例如 2")
    parser.add_argument("--threshold", type=float, default=None,
                        help="检测阈值。不填则使用 0.5 * sin(theta) 作为初步阈值")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda 或 cpu")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    device = torch.device(args.device)

    key = torch_load(args.key, map_location="cpu")

    labels = key["labels"]
    wm_dirs = key["wm_dirs"].float()
    clean_embeddings = key["clean_embeddings"].float()
    watermarked_embeddings = key["watermarked_embeddings"].float()
    theta = float(key["theta"])
    sin_theta = float(key["sin_theta"])

    idx = find_label_index(labels, args.label)

    wi = F.normalize(wm_dirs[idx], p=2, dim=0)
    g_clean = F.normalize(clean_embeddings[idx], p=2, dim=0)
    g_wm = F.normalize(watermarked_embeddings[idx], p=2, dim=0)

    model = load_backbone(args.network, args.weights, device)
    z = extract_embedding(model, args.image, device)

    # 水印检测分数
    s_wm = torch.sum(z * wi).item()

    # 辅助观察项
    cos_to_clean = torch.sum(z * g_clean).item()
    cos_to_wm = torch.sum(z * g_wm).item()

    # 理论上，如果 z 非常接近带水印模板 g_wm，则 s_wm 接近 sin(theta)
    if args.threshold is None:
        threshold = 0.5 * sin_theta
        threshold_source = "default: 0.5 * sin(theta)"
    else:
        threshold = args.threshold
        threshold_source = "user provided"

    detected = s_wm > threshold

    print("========== Watermark Detection ==========")
    print("image:", args.image)
    print("target label:", args.label)
    print("label index:", idx)
    print("theta:", theta)
    print("expected sin(theta):", sin_theta)
    print("threshold:", threshold)
    print("threshold source:", threshold_source)
    print("-----------------------------------------")
    print("watermark score s_wm = <z, w_i>:", s_wm)
    print("cos(z, clean_gallery):", cos_to_clean)
    print("cos(z, watermarked_gallery):", cos_to_wm)
    print("cos(z, wm) - cos(z, clean):", cos_to_wm - cos_to_clean)
    print("-----------------------------------------")
    print("detected:", int(detected))

    if detected:
        print("result: watermark detected")
    else:
        print("result: watermark not detected")


if __name__ == "__main__":
    main()