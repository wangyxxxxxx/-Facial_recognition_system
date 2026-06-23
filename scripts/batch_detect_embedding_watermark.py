import argparse
import csv
import glob
import math
import os
import re
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
    try:
        from backbones import get_model
    except Exception as e:
        print("无法 import backbones.get_model。")
        print("请把 batch_detect_embedding_watermark.py 放到 arcface_torch 目录下运行。")
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
    target_str = str(target_label)

    for i, lab in enumerate(labels):
        if str(lab) == target_str:
            return i

    raise ValueError(
        f"在 watermark_key 的 labels 中找不到 label={target_label}。"
        f"前20个 labels: {labels[:20]}"
    )


def parse_label_from_filename(path):
    """
    默认解析 direct_type1_2.png 里的 2。
    支持：
        direct_type1_2.png
        direct_type2_15.png
        xxx_37.png
    """
    name = os.path.basename(path)

    m = re.search(r"direct_type\d+_(\d+)\.(png|jpg|jpeg|bmp)$", name, re.IGNORECASE)
    if m:
        return m.group(1)

    m = re.search(r"_(\d+)\.(png|jpg|jpeg|bmp)$", name, re.IGNORECASE)
    if m:
        return m.group(1)

    raise ValueError(f"无法从文件名解析 label: {name}，请使用 --label 指定固定 label。")


def collect_images(image_dir, pattern, recursive):
    search_path = os.path.join(image_dir, pattern)

    files = glob.glob(search_path, recursive=recursive)

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = [p for p in files if os.path.splitext(p)[1].lower() in exts]

    def sort_key(p):
        name = os.path.basename(p)
        nums = re.findall(r"\d+", name)
        if nums:
            return int(nums[-1])
        return name

    files = sorted(files, key=sort_key)
    return files


def compute_one(model, image_path, label, key, device, threshold):
    labels = key["labels"]
    wm_dirs = key["wm_dirs"].float()
    clean_embeddings = key["clean_embeddings"].float()
    watermarked_embeddings = key["watermarked_embeddings"].float()

    idx = find_label_index(labels, label)

    wi = F.normalize(wm_dirs[idx], p=2, dim=0)
    g_clean = F.normalize(clean_embeddings[idx], p=2, dim=0)
    g_wm = F.normalize(watermarked_embeddings[idx], p=2, dim=0)

    z = extract_embedding(model, image_path, device)

    s_wm = torch.sum(z * wi).item()
    cos_to_clean = torch.sum(z * g_clean).item()
    cos_to_wm = torch.sum(z * g_wm).item()
    diff = cos_to_wm - cos_to_clean
    detected = int(s_wm > threshold)

    return {
        "image": image_path,
        "filename": os.path.basename(image_path),
        "label": str(label),
        "label_index": idx,
        "s_wm": s_wm,
        "cos_clean": cos_to_clean,
        "cos_wm": cos_to_wm,
        "cos_wm_minus_clean": diff,
        "threshold": threshold,
        "detected": detected,
    }


def mean(xs):
    return sum(xs) / len(xs) if len(xs) > 0 else float("nan")


def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def percentile(xs, q):
    if len(xs) == 0:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def print_summary(rows):
    s_vals = [r["s_wm"] for r in rows]
    clean_vals = [r["cos_clean"] for r in rows]
    wm_vals = [r["cos_wm"] for r in rows]
    diff_vals = [r["cos_wm_minus_clean"] for r in rows]
    det_vals = [r["detected"] for r in rows]

    print()
    print("========== Batch Summary ==========")
    print("num images:", len(rows))
    print("-----------------------------------")
    print("s_wm mean:", mean(s_vals))
    print("s_wm std:", std(s_vals))
    print("s_wm min:", min(s_vals) if s_vals else float("nan"))
    print("s_wm max:", max(s_vals) if s_vals else float("nan"))
    print("s_wm p50:", percentile(s_vals, 50))
    print("s_wm p90:", percentile(s_vals, 90))
    print("s_wm p95:", percentile(s_vals, 95))
    print("s_wm p99:", percentile(s_vals, 99))
    print("-----------------------------------")
    print("cos_clean mean:", mean(clean_vals))
    print("cos_wm mean:", mean(wm_vals))
    print("cos_wm_minus_clean mean:", mean(diff_vals))
    print("-----------------------------------")
    print("detected count:", sum(det_vals))
    print("detected rate:", mean(det_vals))
    print("===================================")


def save_csv(rows, out_csv):
    if not out_csv:
        return

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fieldnames = [
        "image",
        "filename",
        "label",
        "label_index",
        "s_wm",
        "cos_clean",
        "cos_wm",
        "cos_wm_minus_clean",
        "threshold",
        "detected",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("CSV saved to:", out_csv)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, required=True,
                        help="待批量检测的图片目录，例如 batch_results/clean")
    parser.add_argument("--pattern", type=str, default="*.png",
                        help="图片匹配模式，例如 '*.png' 或 '**/*.png'")
    parser.add_argument("--recursive", action="store_true",
                        help="是否递归搜索子目录")

    parser.add_argument("--weights", type=str, required=True,
                        help="ArcFace backbone 权重，例如 output/fei_r50/model.pt")
    parser.add_argument("--network", type=str, default="r50",
                        help="backbone 名称，例如 r50, r100, mobilefacenet")
    parser.add_argument("--key", type=str, required=True,
                        help="watermark_key.pt")
    parser.add_argument("--label", type=str, default=None,
                        help="固定 label。若不填，则自动从文件名 direct_type1_ID.png 解析 ID。")
    parser.add_argument("--threshold", type=float, default=None,
                        help="检测阈值。不填则使用 0.5 * sin(theta)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda 或 cpu")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="输出 CSV 路径，例如 batch_detect_clean_theta090.csv")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    device = torch.device(args.device)

    key = torch_load(args.key, map_location="cpu")
    theta = float(key["theta"])
    sin_theta = float(key.get("sin_theta", math.sin(theta)))

    if args.threshold is None:
        threshold = 0.5 * sin_theta
        threshold_source = "default: 0.5 * sin(theta)"
    else:
        threshold = args.threshold
        threshold_source = "user provided"

    print("========== Batch Watermark Detection ==========")
    print("image_dir:", args.image_dir)
    print("pattern:", args.pattern)
    print("recursive:", args.recursive)
    print("key:", args.key)
    print("theta:", theta)
    print("expected sin(theta):", sin_theta)
    print("threshold:", threshold)
    print("threshold source:", threshold_source)
    print("fixed label:", args.label)
    print("device:", args.device)

    images = collect_images(args.image_dir, args.pattern, args.recursive)

    print("num found images:", len(images))

    if len(images) == 0:
        raise RuntimeError("没有找到待检测图片，请检查 --image_dir 和 --pattern。")

    model = load_backbone(args.network, args.weights, device)

    rows = []
    errors = []

    for i, image_path in enumerate(images, start=1):
        try:
            if args.label is not None:
                label = args.label
            else:
                label = parse_label_from_filename(image_path)

            row = compute_one(
                model=model,
                image_path=image_path,
                label=label,
                key=key,
                device=device,
                threshold=threshold,
            )

            rows.append(row)

            print(
                f"[{i}/{len(images)}] "
                f"label={row['label']} "
                f"s_wm={row['s_wm']:.6f} "
                f"cos_clean={row['cos_clean']:.6f} "
                f"cos_wm={row['cos_wm']:.6f} "
                f"diff={row['cos_wm_minus_clean']:.6f} "
                f"detected={row['detected']} "
                f"{row['filename']}"
            )

        except Exception as e:
            errors.append((image_path, str(e)))
            print(f"[ERROR] {image_path}: {e}")

    if len(rows) > 0:
        print_summary(rows)
        save_csv(rows, args.out_csv)

    if len(errors) > 0:
        print()
        print("========== Errors ==========")
        print("num errors:", len(errors))
        for p, msg in errors[:20]:
            print(p, "=>", msg)


if __name__ == "__main__":
    main()