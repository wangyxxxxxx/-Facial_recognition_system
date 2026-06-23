# -*- coding: utf-8 -*-
"""
score_defense.py

黑盒分数防御模块。

设计目标：
    1. api_server_lab.py 中防御关闭时完全走原来的 /score、/score_batch 算分代码；
    2. 只有管理员开启分数防御后，api_server_lab.py 才调用本文件；
    3. watermark key 加载、内部组件计算、防御公式都放在本文件；
    4. clean_score / wm_dir_score / max_clean_score / wm_gallery_score 只在服务端内部使用，不对外返回。
"""

import math
import os
import random
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from app_cli import extract_embedding, find_label_index


def normalize_bool(value, default_value=False) -> bool:
    """把管理员配置中的 bool / 数字 / 字符串统一转成 bool。"""
    if isinstance(value, bool):
        return bool(value)

    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value or "").strip().lower()

    if text in ("1", "true", "yes", "on", "enable", "enabled", "开启", "打开"):
        return True

    if text in ("0", "false", "no", "off", "disable", "disabled", "关闭", "关"):
        return False

    return bool(default_value)


def clip_value(x: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(x)))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def quantize(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return round(float(x) / step) * step


def _torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _labels_to_list(labels):
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().tolist()
    if isinstance(labels, (list, tuple)):
        return list(labels)
    return list(labels)


# =========================
# 防御公式
# =========================

def defend_score_true_score_wm_guided(
    true_score: float,
    clean_score: Optional[float],
    wm_dir_score: Optional[float],
    max_clean_score: Optional[float],
    wm_gallery_score: Optional[float],

    base_score: float = 0.00,

    wm_center: float = 0.02,
    wm_temp: float = 0.120,
    wm_alpha: float = 0.08,
    wm_linear_alpha: float = 0.02,

    id_center: float = 1.00,
    id_temp: float = 1.00,
    id_beta: float = 0.00,

    any_id_center: float = 1.00,
    any_id_temp: float = 1.00,
    any_id_beta: float = 0.00,

    gallery_center: float = 1.00,
    gallery_temp: float = 1.00,
    gallery_beta: float = 0.00,

    quant_step: float = 0.000001,
    noise_amp: float = 0.0,
    add_random_noise: bool = False,

    score_low: float = -1.0,
    score_high: float = 1.0,
) -> float:
    """true-score-preserving watermark-guided defense。"""
    s_true = float(true_score)
    s_wm = float(wm_dir_score) if wm_dir_score is not None else s_true
    s_id = float(clean_score) if clean_score is not None else s_true
    s_any = float(max_clean_score) if max_clean_score is not None else s_id
    s_gallery = float(wm_gallery_score) if wm_gallery_score is not None else s_true

    wm_reward = sigmoid((s_wm - wm_center) / wm_temp)
    id_penalty = sigmoid((s_id - id_center) / id_temp)
    any_id_penalty = sigmoid((s_any - any_id_center) / any_id_temp)
    gallery_penalty = sigmoid((s_gallery - gallery_center) / gallery_temp)

    public_score = s_true
    public_score += base_score
    public_score += wm_alpha * (wm_reward - 0.5)
    public_score += wm_linear_alpha * s_wm

    public_score -= id_beta * id_penalty
    public_score -= any_id_beta * any_id_penalty
    public_score -= gallery_beta * gallery_penalty

    if add_random_noise and noise_amp > 0:
        public_score += random.uniform(-noise_amp, noise_amp)

    public_score = quantize(public_score, quant_step)
    return clip_value(public_score, score_low, score_high)


def defend_scores_true_score_wm_guided_batch(
    true_scores: Iterable[float],
    clean_scores: Optional[Iterable[float]],
    wm_dir_scores: Optional[Iterable[float]],
    max_clean_scores: Optional[Iterable[float]],
    wm_gallery_scores: Optional[Iterable[float]],

    base_score: float = 0.00,

    wm_center: float = 0.02,
    wm_temp: float = 0.120,
    wm_alpha: float = 0.08,
    wm_linear_alpha: float = 0.02,

    id_center: float = 1.00,
    id_temp: float = 1.00,
    id_beta: float = 0.00,

    any_id_center: float = 1.00,
    any_id_temp: float = 1.00,
    any_id_beta: float = 0.00,

    gallery_center: float = 1.00,
    gallery_temp: float = 1.00,
    gallery_beta: float = 0.00,

    quant_step: float = 0.000001,
    noise_amp: float = 0.0,
    add_random_noise: bool = False,

    score_low: float = -1.0,
    score_high: float = 1.0,
) -> List[float]:
    true_scores = list(true_scores)

    if clean_scores is None:
        clean_scores = true_scores
    else:
        clean_scores = list(clean_scores)

    if wm_dir_scores is None:
        wm_dir_scores = true_scores
    else:
        wm_dir_scores = list(wm_dir_scores)

    if max_clean_scores is None:
        max_clean_scores = clean_scores
    else:
        max_clean_scores = list(max_clean_scores)

    if wm_gallery_scores is None:
        wm_gallery_scores = true_scores
    else:
        wm_gallery_scores = list(wm_gallery_scores)

    out = []
    for s_true, s_id, s_wm, s_any, s_gallery in zip(
        true_scores,
        clean_scores,
        wm_dir_scores,
        max_clean_scores,
        wm_gallery_scores,
    ):
        out.append(
            float(
                defend_score_true_score_wm_guided(
                    true_score=float(s_true),
                    clean_score=s_id,
                    wm_dir_score=s_wm,
                    max_clean_score=s_any,
                    wm_gallery_score=s_gallery,

                    base_score=base_score,

                    wm_center=wm_center,
                    wm_temp=wm_temp,
                    wm_alpha=wm_alpha,
                    wm_linear_alpha=wm_linear_alpha,

                    id_center=id_center,
                    id_temp=id_temp,
                    id_beta=id_beta,

                    any_id_center=any_id_center,
                    any_id_temp=any_id_temp,
                    any_id_beta=any_id_beta,

                    gallery_center=gallery_center,
                    gallery_temp=gallery_temp,
                    gallery_beta=gallery_beta,

                    quant_step=quant_step,
                    noise_amp=noise_amp,
                    add_random_noise=add_random_noise,

                    score_low=score_low,
                    score_high=score_high,
                )
            )
        )

    return out


# =========================
# 默认防御参数
# =========================

BASE_SCORE = 0.00

WM_CENTER = 0.02
WM_TEMP = 0.120
WM_ALPHA = 0.18
WM_LINEAR_ALPHA = 0.18

ID_CENTER = 1.00
ID_TEMP = 1.00
ID_BETA = 0.00

ANY_ID_CENTER = 1.00
ANY_ID_TEMP = 1.00
ANY_ID_BETA = 0.00

GALLERY_CENTER = 1.00
GALLERY_TEMP = 1.00
GALLERY_BETA = 0.00

QUANT_STEP = 0.07
NOISE_AMP = 0.0
ADD_RANDOM_NOISE = False

SCORE_LOW = -1.00
SCORE_HIGH = 1.00


# =========================
# watermark key 和组件计算
# =========================

_WATERMARK_KEY_CACHE: Dict[str, Dict[str, Any]] = {}


def load_watermark_key(key_path: str):
    if not key_path or not os.path.exists(key_path):
        return None

    mtime = os.path.getmtime(key_path)
    cached = _WATERMARK_KEY_CACHE.get(key_path)

    if cached and cached.get("mtime") == mtime:
        return cached.get("key")

    key = _torch_load(key_path, map_location="cpu")

    if not isinstance(key, dict):
        raise ValueError(f"watermark key 必须是 dict，当前类型: {type(key)}")

    required = [
        "labels",
        "wm_dirs",
        "clean_embeddings",
        "watermarked_embeddings",
    ]

    for name in required:
        if name not in key:
            raise ValueError(f"watermark key 缺少字段: {name}")

    key_for_runtime = dict(key)
    key_for_runtime["labels"] = _labels_to_list(key_for_runtime["labels"])
    key_for_runtime["wm_dirs"] = F.normalize(key_for_runtime["wm_dirs"].float().detach().cpu(), p=2, dim=1)
    key_for_runtime["clean_embeddings"] = F.normalize(key_for_runtime["clean_embeddings"].float().detach().cpu(), p=2, dim=1)
    key_for_runtime["watermarked_embeddings"] = F.normalize(key_for_runtime["watermarked_embeddings"].float().detach().cpu(), p=2, dim=1)

    _WATERMARK_KEY_CACHE[key_path] = {
        "mtime": mtime,
        "key": key_for_runtime,
    }

    return key_for_runtime


@torch.no_grad()
def _extract_z(tmp_path: str, engine) -> torch.Tensor:
    z = extract_embedding(engine.model, tmp_path, engine.device)
    return z.detach().cpu().float()


def _defend_score_with_z_and_key(
    z: torch.Tensor,
    target_label: str,
    true_score: float,
    key: dict,
) -> float:
    """
    已有 embedding z 和 watermark key 时，对某个 label 的 true_score 做防御。

    找不到 key 或 label 时回退 true_score，避免影响正常识别流程。
    """
    if key is None:
        return float(true_score)

    try:
        key_idx = find_label_index(key["labels"], target_label)
    except Exception:
        return float(true_score)

    g_clean = key["clean_embeddings"][key_idx]
    w_i = key["wm_dirs"][key_idx]
    g_wm = key["watermarked_embeddings"][key_idx]

    clean_score = float(torch.sum(z * g_clean).item())
    wm_dir_score = float(torch.sum(z * w_i).item())
    wm_gallery_score = float(torch.sum(z * g_wm).item())

    all_clean_scores = torch.matmul(key["clean_embeddings"], z)
    max_clean_score = float(torch.max(all_clean_scores).item())

    return float(
        defend_score_true_score_wm_guided(
            true_score=float(true_score),
            clean_score=clean_score,
            wm_dir_score=wm_dir_score,
            max_clean_score=max_clean_score,
            wm_gallery_score=wm_gallery_score,

            base_score=BASE_SCORE,

            wm_center=WM_CENTER,
            wm_temp=WM_TEMP,
            wm_alpha=WM_ALPHA,
            wm_linear_alpha=WM_LINEAR_ALPHA,

            id_center=ID_CENTER,
            id_temp=ID_TEMP,
            id_beta=ID_BETA,

            any_id_center=ANY_ID_CENTER,
            any_id_temp=ANY_ID_TEMP,
            any_id_beta=ANY_ID_BETA,

            gallery_center=GALLERY_CENTER,
            gallery_temp=GALLERY_TEMP,
            gallery_beta=GALLERY_BETA,

            quant_step=QUANT_STEP,
            noise_amp=NOISE_AMP,
            add_random_noise=ADD_RANDOM_NOISE,

            score_low=SCORE_LOW,
            score_high=SCORE_HIGH,
        )
    )


def defend_score_file(
    tmp_path: str,
    target_label: str,
    gallery: dict,
    engine,
    true_score: float,
    label_index: int,
    key_path: str,
) -> float:
    """单图防御入口。只应该在 api_server_lab.py 已确认“防御开启”后调用。"""
    key = load_watermark_key(key_path)
    if key is None:
        return float(true_score)

    z = _extract_z(tmp_path, engine)
    return _defend_score_with_z_and_key(
        z=z,
        target_label=target_label,
        true_score=true_score,
        key=key,
    )


def defend_score_files(
    tmp_paths: List[str],
    target_label: str,
    gallery: dict,
    engine,
    true_scores: List[float],
    label_index: int,
    key_path: str,
) -> List[float]:
    """批量防御入口。"""
    return [
        defend_score_file(
            tmp_path=p,
            target_label=target_label,
            gallery=gallery,
            engine=engine,
            true_score=s,
            label_index=label_index,
            key_path=key_path,
        )
        for p, s in zip(tmp_paths, true_scores)
    ]


def defend_predict_topk_result(
    tmp_path: str,
    result: dict,
    engine,
    key_path: str,
) -> dict:
    """
    predict 分支专用：
        1. pred_label 和 topk 排序仍由正常原始分数决定；
        2. 只把返回给前端的 top1_cosine / topk[*].cosine 替换成防御后的分数；
        3. watermark key 不存在或 label 找不到时，对应分数回退原始分数。
    """
    key = load_watermark_key(key_path)
    if key is None:
        return result

    z = _extract_z(tmp_path, engine)
    defended = dict(result)
    defended_topk = []

    for item in result.get("topk", []):
        item_out = dict(item)
        label = str(item.get("label"))
        true_score = float(item.get("cosine", 0.0))
        item_out["cosine"] = _defend_score_with_z_and_key(
            z=z,
            target_label=label,
            true_score=true_score,
            key=key,
        )
        defended_topk.append(item_out)

    defended["topk"] = defended_topk

    if defended_topk:
        defended["top1_cosine"] = float(defended_topk[0].get("cosine", result.get("top1_cosine", 0.0)))
    else:
        defended["top1_cosine"] = _defend_score_with_z_and_key(
            z=z,
            target_label=str(result.get("pred_label", "")),
            true_score=float(result.get("top1_cosine", 0.0)),
            key=key,
        )

    return defended
