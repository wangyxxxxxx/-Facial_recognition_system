import argparse
import copy
import math
import os
import torch
import torch.nn.functional as F


def torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.tensor(x)


def is_2d_embedding_tensor(x):
    if not isinstance(x, torch.Tensor):
        return False
    return x.ndim == 2 and x.shape[1] >= 16


def extract_gallery(obj):
    """
    尽量兼容常见 gallery 格式：
    1. Tensor: [N, D]
    2. dict: {"features": Tensor[N,D], "labels": Tensor[N]}
    3. dict: {"embeddings": Tensor[N,D], "labels": Tensor[N]}
    4. tuple/list: (embeddings, labels)
    5. dict: {label: embedding}
    """
    # 情况1：直接是 Tensor[N, D]
    if is_2d_embedding_tensor(obj):
        emb = obj.float()
        labels = list(range(emb.shape[0]))
        spec = {
            "kind": "tensor",
        }
        return emb, labels, spec

    # 情况2：dict
    if isinstance(obj, dict):
        candidate_emb_keys = [
            "embeddings", "embedding", "features", "feature", "feats", "feat",
            "gallery_embeddings", "gallery_features", "gallery_feats",
            "embs", "vectors", "prototypes", "prototype", "centers"
        ]
        candidate_label_keys = [
            "labels", "label", "ids", "id", "targets", "target", "names"
        ]

        emb_key = None
        for k in candidate_emb_keys:
            if k in obj and is_2d_embedding_tensor(to_tensor(obj[k])):
                emb_key = k
                break

        if emb_key is not None:
            emb = to_tensor(obj[emb_key]).float()

            label_key = None
            labels = None
            for k in candidate_label_keys:
                if k in obj:
                    temp = obj[k]
                    if isinstance(temp, torch.Tensor):
                        if temp.ndim == 1 and temp.shape[0] == emb.shape[0]:
                            label_key = k
                            labels = temp.detach().cpu().tolist()
                            break
                    elif isinstance(temp, (list, tuple)):
                        if len(temp) == emb.shape[0]:
                            label_key = k
                            labels = list(temp)
                            break

            if labels is None:
                labels = list(range(emb.shape[0]))

            spec = {
                "kind": "dict_tensor_key",
                "emb_key": emb_key,
                "label_key": label_key,
            }
            return emb, labels, spec

        # 情况5：dict[label] = embedding
        keys = list(obj.keys())
        values = []
        ok = True
        for k in keys:
            v = to_tensor(obj[k])
            if v.ndim == 1 and v.numel() >= 16:
                values.append(v.float())
            else:
                ok = False
                break

        if ok and len(values) > 0:
            emb = torch.stack(values, dim=0)
            labels = keys
            spec = {
                "kind": "dict_by_label",
                "labels": labels,
            }
            return emb, labels, spec

    # 情况3：tuple/list
    if isinstance(obj, (tuple, list)):
        emb_index = None
        for i, item in enumerate(obj):
            item_t = to_tensor(item)
            if is_2d_embedding_tensor(item_t):
                emb_index = i
                break

        if emb_index is not None:
            emb = to_tensor(obj[emb_index]).float()

            labels = None
            label_index = None
            for i, item in enumerate(obj):
                if i == emb_index:
                    continue
                if isinstance(item, torch.Tensor):
                    if item.ndim == 1 and item.shape[0] == emb.shape[0]:
                        labels = item.detach().cpu().tolist()
                        label_index = i
                        break
                elif isinstance(item, (list, tuple)):
                    if len(item) == emb.shape[0]:
                        labels = list(item)
                        label_index = i
                        break

            if labels is None:
                labels = list(range(emb.shape[0]))

            spec = {
                "kind": "seq",
                "emb_index": emb_index,
                "label_index": label_index,
                "is_tuple": isinstance(obj, tuple),
            }
            return emb, labels, spec

    raise ValueError(
        "无法识别 gallery.pt 的结构。请先打印 torch.load(gallery).keys() 或把结构发给我。"
    )


def replace_gallery_embeddings(obj, wm_emb, spec):
    wm_emb = wm_emb.detach().cpu()

    if spec["kind"] == "tensor":
        return wm_emb

    if spec["kind"] == "dict_tensor_key":
        new_obj = copy.deepcopy(obj)
        new_obj[spec["emb_key"]] = wm_emb
        return new_obj

    if spec["kind"] == "dict_by_label":
        new_obj = copy.deepcopy(obj)
        labels = spec["labels"]
        for i, label in enumerate(labels):
            new_obj[label] = wm_emb[i]
        return new_obj

    if spec["kind"] == "seq":
        new_list = list(copy.deepcopy(obj))
        new_list[spec["emb_index"]] = wm_emb
        if spec["is_tuple"]:
            return tuple(new_list)
        return new_list

    raise ValueError("未知 gallery spec。")


def make_numeric_labels_if_possible(labels):
    try:
        return torch.tensor([int(x) for x in labels], dtype=torch.long)
    except Exception:
        return None


def make_watermarked_embeddings(clean_emb, theta, seed):
    """
    clean_emb: [N, D]
    theta: 水印角度强度，单位 rad

    构造：
        w_i = w - (w^T g_i) g_i
        g_i_wm = cos(theta) g_i + sin(theta) w_i
    """
    clean_emb = F.normalize(clean_emb.float(), p=2, dim=1)
    n, d = clean_emb.shape

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    global_w = torch.randn(d, generator=gen)
    global_w = F.normalize(global_w, p=2, dim=0)

    wm_dirs = []
    wm_embs = []

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    for i in range(n):
        g = clean_emb[i]

        # 将全局水印方向正交到当前身份 embedding 的切空间里
        wi = global_w - torch.sum(global_w * g) * g
        wi_norm = torch.norm(wi, p=2)

        # 极低概率：global_w 和 g 太接近，重新生成一个方向
        if wi_norm < 1e-6:
            wi = torch.randn(d, generator=gen)
            wi = wi - torch.sum(wi * g) * g
            wi_norm = torch.norm(wi, p=2)

        wi = wi / wi_norm

        g_wm = cos_t * g + sin_t * wi
        g_wm = F.normalize(g_wm, p=2, dim=0)

        wm_dirs.append(wi)
        wm_embs.append(g_wm)

    wm_dirs = torch.stack(wm_dirs, dim=0)
    wm_embs = torch.stack(wm_embs, dim=0)

    return clean_emb, wm_embs, global_w, wm_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery", type=str, required=True,
                        help="原始 gallery.pt，例如 output/fei_r50_protected/fei_gallery.pt")
    parser.add_argument("--out_gallery", type=str, required=True,
                        help="输出带水印 gallery.pt")
    parser.add_argument("--out_key", type=str, required=True,
                        help="输出 watermark_key.pt，用于后续检测")
    parser.add_argument("--theta", type=float, default=0.10,
                        help="水印角度强度，单位 rad。建议先试 0.05, 0.10, 0.15")
    parser.add_argument("--seed", type=int, default=2026,
                        help="生成全局水印方向的随机种子")
    parser.add_argument("--dry_run", action="store_true",
                        help="只查看 gallery 结构，不保存")
    args = parser.parse_args()

    gallery_obj = torch_load(args.gallery)
    clean_emb, labels, spec = extract_gallery(gallery_obj)

    print("========== Gallery Info ==========")
    print("gallery path:", args.gallery)
    print("format spec:", spec)
    print("embedding shape:", tuple(clean_emb.shape))
    print("num labels:", len(labels))
    print("first 10 labels:", labels[:10])
    print("theta:", args.theta)
    print("seed:", args.seed)
    print("cos(theta):", math.cos(args.theta))
    print("sin(theta):", math.sin(args.theta))

    if args.dry_run:
        print("dry_run enabled. No file saved.")
        return

    clean_emb, wm_emb, global_w, wm_dirs = make_watermarked_embeddings(
        clean_emb=clean_emb,
        theta=args.theta,
        seed=args.seed,
    )

    new_gallery_obj = replace_gallery_embeddings(gallery_obj, wm_emb, spec)

    os.makedirs(os.path.dirname(args.out_gallery) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_key) or ".", exist_ok=True)

    torch.save(new_gallery_obj, args.out_gallery)

    key = {
        "theta": args.theta,
        "seed": args.seed,
        "cos_theta": math.cos(args.theta),
        "sin_theta": math.sin(args.theta),
        "global_w": global_w.cpu(),
        "wm_dirs": wm_dirs.cpu(),
        "clean_embeddings": clean_emb.cpu(),
        "watermarked_embeddings": wm_emb.cpu(),
        "labels": labels,
        "labels_tensor_if_numeric": make_numeric_labels_if_possible(labels),
        "gallery_spec": spec,
        "source_gallery": args.gallery,
        "out_gallery": args.out_gallery,
    }

    torch.save(key, args.out_key)

    print("========== Saved ==========")
    print("watermarked gallery:", args.out_gallery)
    print("watermark key:", args.out_key)

    # 简单检查
    cos_clean_wm = torch.sum(clean_emb * wm_emb, dim=1)
    wm_response = torch.sum(wm_emb * wm_dirs, dim=1)

    print("========== Check ==========")
    print("mean cos(clean, wm):", float(cos_clean_wm.mean()))
    print("expected cos(theta):", math.cos(args.theta))
    print("mean wm response <g_wm, w_i>:", float(wm_response.mean()))
    print("expected sin(theta):", math.sin(args.theta))


if __name__ == "__main__":
    main()