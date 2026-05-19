import csv
import hashlib
import json
import math
import os
import shutil
import uuid
import tempfile
from datetime import datetime, timedelta
from typing import Optional, List

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

from app_cli import (
    ArcFaceCLI,
    load_gallery,
    extract_embedding,
    find_label_index,
)

# 水印生成直接复用用户提供的 make_embedding_watermark_gallery.py
# 请把 make_embedding_watermark_gallery.py 和 api_server_lab.py 放在同一项目根目录。
from make_embedding_watermark_gallery import (
    extract_gallery as wm_extract_gallery,
    make_watermarked_embeddings as wm_make_watermarked_embeddings,
    replace_gallery_embeddings as wm_replace_gallery_embeddings,
    make_numeric_labels_if_possible as wm_make_numeric_labels_if_possible,
)


# ==========================
# 默认路径配置
# ==========================

DEFAULT_WEIGHTS = r"weights\fei_r50-train\model.pt"

# 注意：prior 人脸识别系统不可用，这里不再保留 prior。
GALLERY_PATHS = {
    "clean": r"weights\fei_r50-train\fei_gallery.pt",
    "protected": r"weights\fei_r50_protected\fei_gallery_wm_theta090.pt",
}

DEFAULT_KEY = r"weights\fei_r50_protected\watermark_key_theta090.pt"

# 只允许管理员在 clean / protected 之间切换。
ALLOWED_RUNTIME_GALLERY_MODES = ["clean", "protected"]


# ==========================
# 动态录入自动水印配置
# ==========================
# 录入/删除只操作 clean gallery，然后自动根据 clean gallery 重新生成 protected gallery 和 watermark key。
DYNAMIC_SOURCE_GALLERY_MODE = "clean"
DYNAMIC_PROTECTED_GALLERY_MODE = "protected"

# 与当前 protected 文件名 fei_gallery_wm_theta090.pt / watermark_key_theta090.pt 对应。
AUTO_WATERMARK_THETA = 0.90
AUTO_WATERMARK_SEED = 2026

DEFAULT_NETWORK = "r50"
DEFAULT_DEVICE = "cpu"

# 普通人脸验证阈值，可以按你的实验改
DEFAULT_API_THRESHOLD = 0.30


# ==========================
# 管理员人脸认证配置
# ==========================
# 管理员身份必须存在于 ADMIN_GALLERY_MODE 对应的 gallery 里。
ADMIN_LABEL = "1"
ADMIN_GALLERY_MODE = "clean"
ADMIN_FACE_THRESHOLD = 0.30
ADMIN_TOKEN_TTL_SECONDS = 1800

# 管理员验证通过后生成短期 token，后续录入、删除、模式切换接口必须携带该 token。
ADMIN_TOKENS = {}


# ==========================
# 水印检测配置
# ==========================

# 你前面根据 clean 负样本统计得到的水印检测阈值
DEFAULT_WM_THRESHOLD = 0.085

# score_batch 默认交给管理员配置决定
DEFAULT_BATCH_GALLERY_MODE = "auto"


# ==========================
# 后端调试输出配置
# ==========================
# 开启后，/predict、/both、/admin_verify、/enroll_face 会把识别分数输出到
# PyCharm 控制台和 api_logs/backend.log，方便调试 201/202 这类混淆问题。
DEBUG_RECOGNITION_SCORES = True
DEBUG_TOPK = 10


# ==========================
# 日志和运行配置文件
# ==========================

LOG_DIR = "api_logs"
UPLOAD_LOG_DIR = os.path.join(LOG_DIR, "uploaded_images")

# 新录入相关文件统一保存到这个独立目录
ENROLLED_FACE_DIR = os.path.join(LOG_DIR, "enrolled_faces")
ENROLLED_IMAGE_DIR = os.path.join(ENROLLED_FACE_DIR, "images")
ENROLLED_GALLERY_BACKUP_DIR = os.path.join(ENROLLED_FACE_DIR, "gallery_backups")

SCORE_LOG_CSV = os.path.join(LOG_DIR, "score_logs.csv")
BATCH_SCORE_LOG_CSV = os.path.join(LOG_DIR, "score_batch_logs.csv")
ENROLL_LOG_CSV = os.path.join(LOG_DIR, "enroll_logs.csv")
DELETE_LOG_CSV = os.path.join(LOG_DIR, "delete_logs.csv")

# 管理员设置的人脸识别 gallery / API 分数 gallery 持久化保存到这里
RUNTIME_CONFIG_JSON = os.path.join(LOG_DIR, "runtime_config.json")


os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(UPLOAD_LOG_DIR, exist_ok=True)
os.makedirs(ENROLLED_FACE_DIR, exist_ok=True)
os.makedirs(ENROLLED_IMAGE_DIR, exist_ok=True)
os.makedirs(ENROLLED_GALLERY_BACKUP_DIR, exist_ok=True)


app = FastAPI(title="ArcFace Local Lab API")


# 启动时只加载一次模型
engine = ArcFaceCLI(
    weights=DEFAULT_WEIGHTS,
    network=DEFAULT_NETWORK,
    device=DEFAULT_DEVICE,
)


# gallery 缓存，避免每次请求重复加载
GALLERY_CACHE = {}


# ==========================
# 基础工具函数
# ==========================

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def debug_print(message: str):
    """
    后端调试输出。
    run_app.py 启动 uvicorn 时会把 stdout/stderr 写入 api_logs/backend.log，
    所以这里的 print 既能在控制台看到，也会进入 backend.log。
    """
    if DEBUG_RECOGNITION_SCORES:
        print(f"[{now_str()}] {message}", flush=True)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_upload_file(upload_file: UploadFile) -> str:
    suffix = os.path.splitext(upload_file.filename or "")[-1]
    if suffix == "":
        suffix = ".jpg"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)

    return tmp_path


def save_image_for_log(tmp_path, sha256_value, suffix=".jpg"):
    out_path = os.path.join(UPLOAD_LOG_DIR, f"{sha256_value}{suffix}")
    if not os.path.exists(out_path):
        shutil.copy(tmp_path, out_path)
    return out_path


def safe_filename_part(text: str) -> str:
    """
    将 label / gallery_mode 等转成适合文件名的片段。
    """
    text = str(text).strip()
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    value = "".join(keep).strip("._")
    return value or "unknown"


def save_enrolled_image_for_log(tmp_path, label, sha256_value, suffix=".jpg"):
    """
    新录入的人脸图片统一备份到：
        api_logs/enrolled_faces/images/<label>/时间_label_hash.jpg
    这样不会和普通识别/水印检测上传图片混在 uploaded_images 里。
    """
    safe_label = safe_filename_part(label)
    label_dir = os.path.join(ENROLLED_IMAGE_DIR, safe_label)
    os.makedirs(label_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    short_hash = str(sha256_value)[:12]
    filename = f"{ts}_{safe_label}_{short_hash}{suffix}"
    out_path = os.path.join(label_dir, filename)
    shutil.copy(tmp_path, out_path)
    return out_path


def torch_load_local(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def labels_to_list(labels):
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().tolist()
    if isinstance(labels, (list, tuple)):
        return list(labels)
    return list(labels)


def label_index_or_none(labels, label):
    target = str(label)
    for i, lab in enumerate(labels):
        if str(lab) == target:
            return i
    return None


# ==========================
# 管理员运行模式配置
# ==========================


def default_runtime_config():
    return {
        "gallery_mode": "protected",
        "api_threshold": float(DEFAULT_API_THRESHOLD),
        "watermark_threshold": float(DEFAULT_WM_THRESHOLD),
        "updated_at": "",
        "updated_by": "system",
    }


def normalize_runtime_gallery_mode(value, default_value="protected"):
    value = str(value or "").strip()
    if value in ALLOWED_RUNTIME_GALLERY_MODES:
        return value
    return default_value


def normalize_float(value, default_value):
    try:
        return float(value)
    except Exception:
        return float(default_value)


def load_runtime_config():
    cfg = default_runtime_config()

    if os.path.exists(RUNTIME_CONFIG_JSON):
        try:
            with open(RUNTIME_CONFIG_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                cfg.update(data)
        except Exception as e:
            debug_print(f"[runtime_config] 读取失败，使用默认配置：{e}")

    cfg["gallery_mode"] = normalize_runtime_gallery_mode(
        cfg.get("gallery_mode"),
        default_value="protected",
    )
    cfg["api_threshold"] = normalize_float(
        cfg.get("api_threshold"),
        DEFAULT_API_THRESHOLD,
    )
    cfg["watermark_threshold"] = normalize_float(
        cfg.get("watermark_threshold"),
        DEFAULT_WM_THRESHOLD,
    )

    return cfg


def save_runtime_config():
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(RUNTIME_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(RUNTIME_CONFIG, f, ensure_ascii=False, indent=2)


RUNTIME_CONFIG = load_runtime_config()


def get_runtime_config_public():
    return {
        "gallery_mode": RUNTIME_CONFIG.get("gallery_mode", "protected"),
        "api_threshold": float(RUNTIME_CONFIG.get("api_threshold", DEFAULT_API_THRESHOLD)),
        "watermark_threshold": float(RUNTIME_CONFIG.get("watermark_threshold", DEFAULT_WM_THRESHOLD)),
        "updated_at": RUNTIME_CONFIG.get("updated_at", ""),
        "updated_by": RUNTIME_CONFIG.get("updated_by", ""),
        "allowed_gallery_modes": ALLOWED_RUNTIME_GALLERY_MODES,
    }


def get_runtime_gallery_mode():
    """
    返回当前系统唯一 gallery 模式。

    所有需要 gallery 的接口统一使用这个模式：
        /predict
        /both 中的识别阶段
        /score
        /score_batch

    普通请求传来的 gallery_mode 只作为日志里的 requested_gallery_mode，
    不决定实际使用的 gallery。
    """
    return normalize_runtime_gallery_mode(
        RUNTIME_CONFIG.get("gallery_mode"),
        default_value="protected",
    )


def get_runtime_api_threshold():
    return float(RUNTIME_CONFIG.get("api_threshold", DEFAULT_API_THRESHOLD))


def get_runtime_watermark_threshold():
    return float(RUNTIME_CONFIG.get("watermark_threshold", DEFAULT_WM_THRESHOLD))


def validate_runtime_mode_or_error(mode: str, field_name: str):
    mode = str(mode or "").strip()
    if mode not in ALLOWED_RUNTIME_GALLERY_MODES:
        raise ValueError(
            f"{field_name}={mode} 不合法，可选值: {ALLOWED_RUNTIME_GALLERY_MODES}"
        )
    return mode


def validate_threshold_or_error(value, field_name: str):
    try:
        value = float(value)
    except Exception:
        raise ValueError(f"{field_name} 必须是数字")

    if value < -1.0 or value > 1.0:
        raise ValueError(f"{field_name} 必须在 -1.0 到 1.0 之间")

    return value


# ==========================
# Gallery 读写
# ==========================

def get_gallery(gallery_mode):
    if gallery_mode not in GALLERY_PATHS:
        raise ValueError(
            f"未知 gallery_mode={gallery_mode}，可选值: {list(GALLERY_PATHS.keys())}"
        )

    if gallery_mode not in GALLERY_CACHE:
        gallery_path = GALLERY_PATHS[gallery_mode]
        labels, prototypes = load_gallery(gallery_path)
        GALLERY_CACHE[gallery_mode] = {
            "path": gallery_path,
            "labels": labels,
            "prototypes": prototypes,
        }

    return GALLERY_CACHE[gallery_mode]


def load_gallery_for_write(gallery_path):
    """
    读取 gallery 原始对象，用于追加/覆盖注册模板。
    兼容本项目常见格式：
        {"labels": [...], "prototypes": Tensor[N,D]}
        {"labels": [...], "features": Tensor[N,D]}
        {"labels": [...], "embeddings": Tensor[N,D]}
        Tensor[N,D]
    """
    obj = torch_load_local(gallery_path, map_location="cpu")

    if isinstance(obj, dict):
        if "prototypes" in obj:
            emb_key = "prototypes"
        elif "features" in obj:
            emb_key = "features"
        elif "embeddings" in obj:
            emb_key = "embeddings"
        else:
            raise ValueError(f"无法识别 gallery keys: {obj.keys()}")

        prototypes = obj[emb_key].float().detach().cpu()
        labels = labels_to_list(obj.get("labels", list(range(prototypes.shape[0]))))
        return obj, emb_key, labels, prototypes

    if isinstance(obj, torch.Tensor):
        prototypes = obj.float().detach().cpu()
        labels = list(range(prototypes.shape[0]))
        new_obj = {
            "labels": labels,
            "prototypes": prototypes,
        }
        return new_obj, "prototypes", labels, prototypes

    raise ValueError(f"不支持的 gallery 类型: {type(obj)}")


def save_gallery_for_write(gallery_path, obj, emb_key, labels, prototypes):
    prototypes = F.normalize(prototypes.float().detach().cpu(), p=2, dim=1)

    if not isinstance(obj, dict):
        obj = {}

    obj["labels"] = labels
    obj[emb_key] = prototypes

    torch.save(obj, gallery_path)


def backup_gallery_file(gallery_path):
    """
    gallery 自动备份统一放到 api_logs/enrolled_faces/gallery_backups/。
    不再散落到 weights 目录旁边。
    """
    if not os.path.exists(gallery_path):
        return ""

    os.makedirs(ENROLLED_GALLERY_BACKUP_DIR, exist_ok=True)
    base = os.path.basename(gallery_path)
    base = safe_filename_part(base)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(ENROLLED_GALLERY_BACKUP_DIR, f"{base}.bak_{ts}")
    shutil.copy2(gallery_path, backup_path)
    return backup_path


# ==========================
# CSV 日志
# ==========================

def append_score_log(row):
    file_exists = os.path.exists(SCORE_LOG_CSV)

    fieldnames = [
        "time",
        "endpoint",
        "requested_gallery_mode",
        "gallery_mode",
        "gallery_path",
        "target_label",
        "label_index",
        "score",
        "api_threshold",
        "success",
        "image_sha256",
        "saved_image_path",
        "original_filename",
    ]

    with open(SCORE_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def append_batch_score_log(row):
    file_exists = os.path.exists(BATCH_SCORE_LOG_CSV)

    fieldnames = [
        "time",
        "endpoint",
        "requested_gallery_mode",
        "gallery_mode",
        "gallery_path",
        "target_id",
        "label_index",
        "num_images",
        "score_mean",
        "score_min",
        "score_max",
        "api_threshold",
    ]

    with open(BATCH_SCORE_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def append_enroll_log(row):
    file_exists = os.path.exists(ENROLL_LOG_CSV)

    fieldnames = [
        "time",
        "endpoint",
        "action",
        "gallery_mode",
        "gallery_path",
        "label",
        "label_index",
        "num_identities",
        "overwrite",
        "image_sha256",
        "saved_image_path",
        "backup_path",
        "original_filename",
        "admin_label",
    ]

    with open(ENROLL_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def append_delete_log(row):
    file_exists = os.path.exists(DELETE_LOG_CSV)

    fieldnames = [
        "time",
        "endpoint",
        "action",
        "gallery_mode",
        "gallery_path",
        "label",
        "deleted_index",
        "num_identities_before",
        "num_identities_after",
        "backup_path",
        "admin_label",
    ]

    with open(DELETE_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ==========================
# 管理员 token
# ==========================

def create_admin_token(admin_label: str):
    token = uuid.uuid4().hex
    expire_at = datetime.now() + timedelta(seconds=ADMIN_TOKEN_TTL_SECONDS)
    ADMIN_TOKENS[token] = {
        "admin_label": str(admin_label),
        "expire_at": expire_at,
    }
    return token, expire_at


def clean_expired_admin_tokens():
    now = datetime.now()
    expired = []
    for token, info in ADMIN_TOKENS.items():
        expire_at = info.get("expire_at")
        if expire_at is None or expire_at < now:
            expired.append(token)

    for token in expired:
        ADMIN_TOKENS.pop(token, None)


def verify_admin_token(admin_token: str):
    clean_expired_admin_tokens()

    if not admin_token:
        return False, "缺少管理员 token"

    info = ADMIN_TOKENS.get(admin_token)
    if not info:
        return False, "管理员 token 无效或已过期"

    return True, info


# ==========================
# 自动生成 protected gallery 和 watermark key
# ==========================

def regenerate_protected_gallery_from_clean(reason: str = "manual"):
    """
    以 clean gallery 为唯一源数据，自动重新生成 protected gallery 和 watermark key。

    动态录入/删除后调用此函数，保证：
        clean gallery：保存原始注册 embedding；
        protected gallery：由 clean gallery 自动加水印得到；
        watermark key：与 protected gallery 的 labels / embedding 对齐。
    """
    clean_path = GALLERY_PATHS[DYNAMIC_SOURCE_GALLERY_MODE]
    protected_path = GALLERY_PATHS[DYNAMIC_PROTECTED_GALLERY_MODE]
    key_path = DEFAULT_KEY

    clean_obj = torch_load_local(clean_path, map_location="cpu")
    clean_emb, labels, spec = wm_extract_gallery(clean_obj)

    clean_emb, wm_emb, global_w, wm_dirs = wm_make_watermarked_embeddings(
        clean_emb=clean_emb,
        theta=AUTO_WATERMARK_THETA,
        seed=AUTO_WATERMARK_SEED,
    )

    protected_gallery_backup_path = backup_gallery_file(protected_path)
    watermark_key_backup_path = backup_gallery_file(key_path)

    protected_obj = wm_replace_gallery_embeddings(clean_obj, wm_emb, spec)
    os.makedirs(os.path.dirname(protected_path) or ".", exist_ok=True)
    torch.save(protected_obj, protected_path)

    key = {
        "theta": AUTO_WATERMARK_THETA,
        "seed": AUTO_WATERMARK_SEED,
        "cos_theta": math.cos(AUTO_WATERMARK_THETA),
        "sin_theta": math.sin(AUTO_WATERMARK_THETA),
        "global_w": global_w.cpu(),
        "wm_dirs": wm_dirs.cpu(),
        "clean_embeddings": clean_emb.cpu(),
        "watermarked_embeddings": wm_emb.cpu(),
        "labels": list(labels),
        "labels_tensor_if_numeric": wm_make_numeric_labels_if_possible(labels),
        "gallery_spec": spec,
        "source_gallery": clean_path,
        "out_gallery": protected_path,
        "reason": reason,
        "updated_at": now_str(),
    }
    os.makedirs(os.path.dirname(key_path) or ".", exist_ok=True)
    torch.save(key, key_path)

    GALLERY_CACHE.pop(DYNAMIC_SOURCE_GALLERY_MODE, None)
    GALLERY_CACHE.pop(DYNAMIC_PROTECTED_GALLERY_MODE, None)

    debug_print(
        f"[auto_watermark] reason={reason} source_script=make_embedding_watermark_gallery.py "
        f"clean={clean_path} protected={protected_path} key={key_path} "
        f"theta={AUTO_WATERMARK_THETA} seed={AUTO_WATERMARK_SEED} num_identities={len(labels)}"
    )

    return {
        "clean_gallery_path": clean_path,
        "protected_gallery_path": protected_path,
        "watermark_key_path": key_path,
        "protected_gallery_backup_path": protected_gallery_backup_path,
        "watermark_key_backup_path": watermark_key_backup_path,
        "theta": AUTO_WATERMARK_THETA,
        "seed": AUTO_WATERMARK_SEED,
        "num_identities": len(labels),
    }


# ==========================
# 识别调试工具
# ==========================

def format_topk_for_debug(topk_items):
    parts = []
    for rank, item in enumerate(topk_items, start=1):
        label = item.get("label", "未知")
        cosine = float(item.get("cosine", 0.0))
        parts.append(f"#{rank}: label={label}, cosine={cosine:.6f}")
    return " | ".join(parts)


def debug_predict_result(endpoint: str, result: dict, gallery_mode: str, gallery_path: str, image_name: str = ""):
    if not DEBUG_RECOGNITION_SCORES:
        return

    pred_label = str(result.get("pred_label", "未知"))
    top1_cosine = float(result.get("top1_cosine", 0.0))
    topk_items = result.get("topk", [])[:DEBUG_TOPK]

    debug_print(
        f"[{endpoint}] image={image_name} gallery_mode={gallery_mode} "
        f"gallery_path={gallery_path} pred_label={pred_label} top1_cosine={top1_cosine:.6f}"
    )

    if topk_items:
        debug_print(f"[{endpoint}] Top-{len(topk_items)} scores: {format_topk_for_debug(topk_items)}")

    if len(topk_items) >= 2:
        margin = float(topk_items[0].get("cosine", 0.0)) - float(topk_items[1].get("cosine", 0.0))
        debug_print(f"[{endpoint}] top1-top2 margin={margin:.6f}")


def compute_topk_against_gallery(query_emb: torch.Tensor, labels, prototypes: torch.Tensor, topk: int = DEBUG_TOPK):
    """
    用于录入后调试：查看新录入 embedding 与当前 gallery 里哪些身份最接近。
    """
    if prototypes is None or prototypes.numel() == 0:
        return []

    z = F.normalize(query_emb.float().detach().cpu(), p=2, dim=0)
    p = F.normalize(prototypes.float().detach().cpu(), p=2, dim=1)
    scores = torch.matmul(p, z)
    k = min(int(topk), int(scores.numel()))
    values, indices = torch.topk(scores, k=k)

    items = []
    for value, index in zip(values, indices):
        idx = int(index.item())
        items.append(
            {
                "label": str(labels[idx]),
                "index": idx,
                "cosine": float(value.item()),
            }
        )
    return items


def score_image_file_against_target(tmp_path: str, target_emb: torch.Tensor) -> float:
    """
    对单张临时图片提取 embedding，然后和目标 gallery embedding 计算余弦相似度。
    """
    z = extract_embedding(engine.model, tmp_path, engine.device)
    score = torch.sum(z * target_emb).item()
    return float(score)


# ==========================
# 基础接口
# ==========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEFAULT_DEVICE,
        "weights": DEFAULT_WEIGHTS,
        "gallery_modes": GALLERY_PATHS,
        "allowed_runtime_gallery_modes": ALLOWED_RUNTIME_GALLERY_MODES,
        "runtime_config": get_runtime_config_public(),
        "runtime_config_json": RUNTIME_CONFIG_JSON,
        "watermark_key": DEFAULT_KEY,
        "score_log_csv": SCORE_LOG_CSV,
        "score_batch_log_csv": BATCH_SCORE_LOG_CSV,
        "enroll_log_csv": ENROLL_LOG_CSV,
        "delete_log_csv": DELETE_LOG_CSV,
        "enrolled_face_dir": ENROLLED_FACE_DIR,
        "enrolled_image_dir": ENROLLED_IMAGE_DIR,
        "enrolled_gallery_backup_dir": ENROLLED_GALLERY_BACKUP_DIR,
        "dynamic_source_gallery_mode": DYNAMIC_SOURCE_GALLERY_MODE,
        "dynamic_protected_gallery_mode": DYNAMIC_PROTECTED_GALLERY_MODE,
        "auto_watermark_theta": AUTO_WATERMARK_THETA,
        "auto_watermark_seed": AUTO_WATERMARK_SEED,
        "score_batch_default_gallery_mode": DEFAULT_BATCH_GALLERY_MODE,
        "admin_label": ADMIN_LABEL,
        "admin_gallery_mode": ADMIN_GALLERY_MODE,
        "admin_face_threshold": ADMIN_FACE_THRESHOLD,
        "admin_token_ttl_seconds": ADMIN_TOKEN_TTL_SECONDS,
        "debug_recognition_scores": DEBUG_RECOGNITION_SCORES,
        "debug_topk": DEBUG_TOPK,
    }


@app.get("/admin_runtime_config")
def get_admin_runtime_config():
    """
    查看当前系统运行模式和阈值。
    读取配置不需要管理员 token；修改配置必须管理员 token。
    """
    return {
        "success": True,
        **get_runtime_config_public(),
    }


@app.post("/admin_runtime_config")
async def update_admin_runtime_config(
    admin_token: str = Form(...),
    gallery_mode: str = Form(...),
    api_threshold: float = Form(...),
    watermark_threshold: float = Form(...),
):
    """
    管理员运行配置接口。

    gallery_mode:
        系统唯一图库模式，控制 /predict、/both、/score、/score_batch。

    api_threshold:
        控制 /score 和 /score_batch 的验证阈值。

    watermark_threshold:
        控制 /detect_watermark 和 /both 的水印检测阈值。
    """
    ok, token_info = verify_admin_token(admin_token)
    if not ok:
        return JSONResponse(
            status_code=403,
            content={
                "success": False,
                "error": token_info,
            },
        )

    try:
        gallery_mode = validate_runtime_mode_or_error(
            gallery_mode,
            "gallery_mode",
        )
        api_threshold = validate_threshold_or_error(
            api_threshold,
            "api_threshold",
        )
        watermark_threshold = validate_threshold_or_error(
            watermark_threshold,
            "watermark_threshold",
        )

        global RUNTIME_CONFIG

        RUNTIME_CONFIG["gallery_mode"] = gallery_mode
        RUNTIME_CONFIG["api_threshold"] = float(api_threshold)
        RUNTIME_CONFIG["watermark_threshold"] = float(watermark_threshold)
        RUNTIME_CONFIG["updated_at"] = now_str()
        RUNTIME_CONFIG["updated_by"] = str(token_info.get("admin_label", ""))

        save_runtime_config()

        debug_print(
            f"[/admin_runtime_config] updated_by={token_info.get('admin_label', '')} "
            f"gallery_mode={gallery_mode} "
            f"api_threshold={float(api_threshold):.6f} "
            f"watermark_threshold={float(watermark_threshold):.6f}"
        )

        return {
            "success": True,
            **get_runtime_config_public(),
        }

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
                "allowed_gallery_modes": ALLOWED_RUNTIME_GALLERY_MODES,
            },
        )


# ==========================
# 管理员验证、录入、删除
# ==========================

@app.post("/admin_verify")
async def admin_verify(
    image: UploadFile = File(...),
):
    """
    管理员人脸认证接口：
    前端上传当前选择的人脸图片，后端判断它是否属于固定管理员身份 ADMIN_LABEL。

    注意：管理员身份、gallery 和阈值都放在后端，前端不能决定谁是管理员。
    """
    tmp_path = None

    try:
        tmp_path = save_upload_file(image)

        gallery = get_gallery(ADMIN_GALLERY_MODE)
        labels = gallery["labels"]
        prototypes = gallery["prototypes"]

        idx = find_label_index(labels, ADMIN_LABEL)
        admin_emb = prototypes[idx]

        score = score_image_file_against_target(tmp_path, admin_emb)
        verified = bool(score >= ADMIN_FACE_THRESHOLD)

        debug_print(
            f"[/admin_verify] image={image.filename or ''} admin_label={ADMIN_LABEL} "
            f"gallery_mode={ADMIN_GALLERY_MODE} score={float(score):.6f} "
            f"threshold={float(ADMIN_FACE_THRESHOLD):.6f} verified={int(verified)}"
        )

        admin_token = ""
        token_expires_at = ""
        if verified:
            admin_token, expire_at = create_admin_token(ADMIN_LABEL)
            token_expires_at = expire_at.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "success": True,
            "verified": verified,
            "admin_label": str(ADMIN_LABEL),
            "label_index": idx,
            "score": float(score),
            "threshold": float(ADMIN_FACE_THRESHOLD),
            "gallery_mode": ADMIN_GALLERY_MODE,
            "gallery_path": gallery["path"],
            "admin_token": admin_token,
            "token_expires_at": token_expires_at,
            "token_ttl_seconds": ADMIN_TOKEN_TTL_SECONDS,
            "runtime_config": get_runtime_config_public(),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "verified": False,
                "error": str(e),
            },
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/enroll_face")
async def enroll_face(
    image: UploadFile = File(...),
    label: str = Form(...),
    overwrite: bool = Form(False),
    admin_token: str = Form(...),
    save_request_image: bool = Form(True),
):
    """
    录入/更新人脸注册模板：
    管理员验证通过后，前端上传新用户图片和身份 label。

    当前策略：
    - 前端不再选择 clean / protected；
    - 后端只把新身份写入 clean gallery；
    - 写入 clean 后，自动根据 clean gallery 重新生成 protected gallery 和 watermark key；
    - 因此主界面继续使用 protected gallery 识别时，也能识别新录入身份。
    """
    ok, token_info = verify_admin_token(admin_token)
    if not ok:
        return JSONResponse(
            status_code=403,
            content={
                "success": False,
                "error": token_info,
            },
        )

    if not label or str(label).strip() == "":
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "录入身份 label 不能为空",
            },
        )

    label = str(label).strip()
    gallery_mode = DYNAMIC_SOURCE_GALLERY_MODE
    tmp_path = None

    try:
        tmp_path = save_upload_file(image)

        gallery_path = GALLERY_PATHS[gallery_mode]
        obj, emb_key, labels, prototypes = load_gallery_for_write(gallery_path)

        new_emb = extract_embedding(engine.model, tmp_path, engine.device).detach().cpu()
        new_emb = F.normalize(new_emb.float(), p=2, dim=0)

        existing_idx = label_index_or_none(labels, label)

        if existing_idx is not None and not overwrite:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"身份 label={label} 已存在。如需覆盖，请勾选覆盖已有身份。",
                    "label": label,
                    "existing_index": existing_idx,
                },
            )

        clean_backup_path = backup_gallery_file(gallery_path)

        if existing_idx is not None:
            prototypes[existing_idx] = new_emb
            label_index = existing_idx
            action = "update"
        else:
            labels.append(label)
            prototypes = torch.cat([prototypes, new_emb.unsqueeze(0)], dim=0)
            label_index = len(labels) - 1
            action = "insert"

        enroll_topk = compute_topk_against_gallery(new_emb, labels, prototypes, topk=DEBUG_TOPK)
        debug_print(
            f"[/enroll_face] action={action} label={label} label_index={label_index} "
            f"source_gallery_mode={gallery_mode} clean_gallery_path={gallery_path} "
            f"overwrite={int(bool(overwrite))}"
        )
        if enroll_topk:
            debug_print(f"[/enroll_face] New embedding nearest Top-{len(enroll_topk)}: {format_topk_for_debug(enroll_topk)}")

        save_gallery_for_write(gallery_path, obj, emb_key, labels, prototypes)

        # 先清空 clean 缓存，再自动生成 protected gallery 和 watermark key。
        GALLERY_CACHE.pop(gallery_mode, None)
        auto_wm = regenerate_protected_gallery_from_clean(reason=f"enroll_{action}_{label}")

        sha256_value = file_sha256(tmp_path)
        suffix = os.path.splitext(image.filename or "")[-1]
        if suffix == "":
            suffix = ".jpg"

        saved_image_path = ""
        if save_request_image:
            saved_image_path = save_enrolled_image_for_log(tmp_path, label, sha256_value, suffix=suffix)

        append_enroll_log(
            {
                "time": now_str(),
                "endpoint": "/enroll_face",
                "action": action,
                "gallery_mode": gallery_mode,
                "gallery_path": gallery_path,
                "label": label,
                "label_index": label_index,
                "num_identities": len(labels),
                "overwrite": int(bool(overwrite)),
                "image_sha256": sha256_value,
                "saved_image_path": saved_image_path,
                "backup_path": clean_backup_path,
                "original_filename": image.filename or "",
                "admin_label": token_info.get("admin_label", ""),
            }
        )

        warning = (
            "已录入到 clean gallery，并已自动重新生成 protected gallery 和 watermark key。"
            "主界面使用 protected gallery 识别时会自动包含该新身份。"
        )

        return {
            "success": True,
            "action": action,
            "label": label,
            "label_index": label_index,
            "gallery_mode": gallery_mode,
            "gallery_path": gallery_path,
            "source_gallery_mode": DYNAMIC_SOURCE_GALLERY_MODE,
            "source_gallery_path": gallery_path,
            "protected_gallery_mode": DYNAMIC_PROTECTED_GALLERY_MODE,
            "protected_gallery_path": auto_wm.get("protected_gallery_path", ""),
            "watermark_key_path": auto_wm.get("watermark_key_path", ""),
            "num_identities": len(labels),
            "overwrite": bool(overwrite),
            "backup_path": clean_backup_path,
            "auto_watermark": auto_wm,
            "image_sha256": sha256_value,
            "saved_image_path": saved_image_path,
            "warning": warning,
            "runtime_config": get_runtime_config_public(),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/delete_face")
async def delete_face(
    label: str = Form(...),
    admin_token: str = Form(...),
):
    """
    删除已录入/已存在的人脸注册模板。

    当前策略：
    - 前端不再选择 clean / protected；
    - 后端只从 clean gallery 删除；
    - 删除后自动根据 clean gallery 重新生成 protected gallery 和 watermark key；
    - 不删除模型权重、历史日志或已备份图片。
    """
    ok, token_info = verify_admin_token(admin_token)
    if not ok:
        return JSONResponse(
            status_code=403,
            content={
                "success": False,
                "error": token_info,
            },
        )

    if not label or str(label).strip() == "":
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "删除身份 label 不能为空",
            },
        )

    label = str(label).strip()
    gallery_mode = DYNAMIC_SOURCE_GALLERY_MODE

    try:
        # 防止误删管理员本人的注册模板，导致后续无法进入授权页面。
        if gallery_mode == ADMIN_GALLERY_MODE and str(label) == str(ADMIN_LABEL):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"禁止删除当前管理员身份 label={ADMIN_LABEL}。如确需更换管理员，请先修改后端 ADMIN_LABEL/ADMIN_GALLERY_MODE。",
                },
            )

        gallery_path = GALLERY_PATHS[gallery_mode]
        obj, emb_key, labels, prototypes = load_gallery_for_write(gallery_path)

        delete_idx = label_index_or_none(labels, label)
        if delete_idx is None:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "error": f"在 clean gallery 中找不到身份 label={label}",
                    "label": label,
                    "gallery_mode": gallery_mode,
                    "gallery_path": gallery_path,
                },
            )

        num_before = len(labels)
        clean_backup_path = backup_gallery_file(gallery_path)

        new_labels = [lab for i, lab in enumerate(labels) if i != delete_idx]
        if prototypes.shape[0] != len(labels):
            raise ValueError(
                f"gallery labels 数量({len(labels)}) 与 prototypes 数量({prototypes.shape[0]}) 不一致，无法安全删除。"
            )

        keep_indices = [i for i in range(prototypes.shape[0]) if i != delete_idx]
        if keep_indices:
            new_prototypes = prototypes[keep_indices]
        else:
            new_prototypes = prototypes[:0]

        save_gallery_for_write(gallery_path, obj, emb_key, new_labels, new_prototypes)

        GALLERY_CACHE.pop(gallery_mode, None)
        auto_wm = regenerate_protected_gallery_from_clean(reason=f"delete_{label}")

        debug_print(
            f"[/delete_face] label={label} deleted_index={delete_idx} source_gallery_mode={gallery_mode} "
            f"clean_gallery_path={gallery_path} num_before={num_before} num_after={len(new_labels)}"
        )

        append_delete_log(
            {
                "time": now_str(),
                "endpoint": "/delete_face",
                "action": "delete",
                "gallery_mode": gallery_mode,
                "gallery_path": gallery_path,
                "label": label,
                "deleted_index": delete_idx,
                "num_identities_before": num_before,
                "num_identities_after": len(new_labels),
                "backup_path": clean_backup_path,
                "admin_label": token_info.get("admin_label", ""),
            }
        )

        return {
            "success": True,
            "action": "delete",
            "label": label,
            "deleted_index": delete_idx,
            "gallery_mode": gallery_mode,
            "gallery_path": gallery_path,
            "source_gallery_mode": DYNAMIC_SOURCE_GALLERY_MODE,
            "source_gallery_path": gallery_path,
            "protected_gallery_mode": DYNAMIC_PROTECTED_GALLERY_MODE,
            "protected_gallery_path": auto_wm.get("protected_gallery_path", ""),
            "watermark_key_path": auto_wm.get("watermark_key_path", ""),
            "num_identities_before": num_before,
            "num_identities_after": len(new_labels),
            "backup_path": clean_backup_path,
            "auto_watermark": auto_wm,
            "runtime_config": get_runtime_config_public(),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )


# ==========================
# 黑盒分数接口
# ==========================

@app.post("/score")
async def score_api(
    image: UploadFile = File(...),
    target_label: str = Form(...),
    gallery_mode: str = Form("auto"),
    api_threshold: float = Form(DEFAULT_API_THRESHOLD),
    save_request_image: bool = Form(True),
):
    """
    单图黑盒分数接口：
    输入图片 + 目标身份 label
    返回该图片与目标身份注册 embedding 的余弦相似度。

    注意：
    普通请求传来的 gallery_mode 不决定实际使用的 gallery。
    实际使用 clean 还是 protected，由管理员在 /admin_runtime_config 中的 gallery_mode 统一设置。
    """
    tmp_path = None
    requested_gallery_mode = str(gallery_mode or "auto")

    try:
        tmp_path = save_upload_file(image)

        actual_gallery_mode = get_runtime_gallery_mode()
        gallery = get_gallery(actual_gallery_mode)
        labels = gallery["labels"]
        prototypes = gallery["prototypes"]

        idx = find_label_index(labels, target_label)
        target_emb = prototypes[idx]

        score = score_image_file_against_target(tmp_path, target_emb)
        success = bool(score >= api_threshold)

        debug_print(
            f"[/score] image={image.filename or ''} target_label={target_label} "
            f"requested_gallery_mode={requested_gallery_mode} actual_gallery_mode={actual_gallery_mode} "
            f"score={float(score):.6f} threshold={float(api_threshold):.6f} verified={int(success)}"
        )

        sha256_value = file_sha256(tmp_path)
        suffix = os.path.splitext(image.filename or "")[-1]
        if suffix == "":
            suffix = ".jpg"

        saved_image_path = ""
        if save_request_image:
            saved_image_path = save_image_for_log(tmp_path, sha256_value, suffix=suffix)

        log_row = {
            "time": now_str(),
            "endpoint": "/score",
            "requested_gallery_mode": requested_gallery_mode,
            "gallery_mode": actual_gallery_mode,
            "gallery_path": gallery["path"],
            "target_label": str(target_label),
            "label_index": idx,
            "score": score,
            "api_threshold": api_threshold,
            "success": int(success),
            "image_sha256": sha256_value,
            "saved_image_path": saved_image_path,
            "original_filename": image.filename or "",
        }

        append_score_log(log_row)

        return {
            "success": True,
            "requested_gallery_mode": requested_gallery_mode,
            "gallery_mode": actual_gallery_mode,
            "gallery_path": gallery["path"],
            "target_label": str(target_label),
            "label_index": idx,
            "score": score,
            "api_threshold": api_threshold,
            "verified": success,
            "image_sha256": sha256_value,
            "runtime_config": get_runtime_config_public(),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/score_batch")
async def score_batch_api(
    files: List[UploadFile] = File(...),
    target_id: str = Form(...),
    api_threshold: float = Form(DEFAULT_API_THRESHOLD),
    gallery_mode: str = Form(DEFAULT_BATCH_GALLERY_MODE),
):
    """
    批量黑盒分数接口。

    注意：
    普通请求传来的 gallery_mode 不决定实际使用的 gallery。
    实际使用 clean 还是 protected，由管理员在 /admin_runtime_config 中的 gallery_mode 统一设置。
    """
    tmp_paths = []
    requested_gallery_mode = str(gallery_mode or "auto")

    try:
        actual_gallery_mode = get_runtime_gallery_mode()
        gallery = get_gallery(actual_gallery_mode)
        labels = gallery["labels"]
        prototypes = gallery["prototypes"]

        idx = find_label_index(labels, target_id)
        target_emb = prototypes[idx]

        scores = []

        for upload_file in files:
            tmp_path = save_upload_file(upload_file)
            tmp_paths.append(tmp_path)

            score = score_image_file_against_target(tmp_path, target_emb)
            scores.append(float(score))

        if len(scores) > 0:
            score_mean = float(sum(scores) / len(scores))
            score_min = float(min(scores))
            score_max = float(max(scores))
        else:
            score_mean = 0.0
            score_min = 0.0
            score_max = 0.0

        debug_print(
            f"[/score_batch] target_id={target_id} requested_gallery_mode={requested_gallery_mode} "
            f"actual_gallery_mode={actual_gallery_mode} num_images={len(scores)} "
            f"score_mean={score_mean:.6f} score_min={score_min:.6f} "
            f"score_max={score_max:.6f} threshold={float(api_threshold):.6f}"
        )

        append_batch_score_log(
            {
                "time": now_str(),
                "endpoint": "/score_batch",
                "requested_gallery_mode": requested_gallery_mode,
                "gallery_mode": actual_gallery_mode,
                "gallery_path": gallery["path"],
                "target_id": str(target_id),
                "label_index": idx,
                "num_images": len(scores),
                "score_mean": score_mean,
                "score_min": score_min,
                "score_max": score_max,
                "api_threshold": api_threshold,
            }
        )

        return {
            "success": True,
            "target_id": str(target_id),
            "label_index": idx,
            "requested_gallery_mode": requested_gallery_mode,
            "gallery_mode": actual_gallery_mode,
            "gallery_path": gallery["path"],
            "scores": scores,
            "api_threshold": api_threshold,
            "runtime_config": get_runtime_config_public(),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )

    finally:
        for p in tmp_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# ==========================
# 普通识别和水印检测
# ==========================

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    gallery_mode: str = Form("auto"),
    topk: int = Form(5),
):
    """
    普通识别接口：
    输入图片，返回 Top-K 身份。

    注意：
    普通请求传来的 gallery_mode 不决定实际使用的 gallery。
    实际使用 clean 还是 protected，由管理员在 /admin_runtime_config 中的 gallery_mode 统一设置。
    """
    tmp_path = None
    requested_gallery_mode = str(gallery_mode or "auto")

    try:
        tmp_path = save_upload_file(image)

        actual_gallery_mode = get_runtime_gallery_mode()
        gallery = get_gallery(actual_gallery_mode)

        debug_topk = max(int(topk), int(DEBUG_TOPK))
        result = engine.predict(
            image_path=tmp_path,
            gallery_path=gallery["path"],
            topk=debug_topk,
        )

        debug_predict_result(
            endpoint="/predict",
            result=result,
            gallery_mode=actual_gallery_mode,
            gallery_path=gallery["path"],
            image_name=image.filename or "",
        )

        return {
            "success": True,
            "requested_gallery_mode": requested_gallery_mode,
            "gallery_mode": actual_gallery_mode,
            "gallery_path": gallery["path"],
            "pred_label": str(result["pred_label"]),
            "top1_cosine": result["top1_cosine"],
            "topk": [
                {
                    "label": str(item["label"]),
                    "cosine": item["cosine"],
                }
                for item in result["topk"]
            ],
            "runtime_config": get_runtime_config_public(),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/detect_watermark")
async def detect_watermark(
    image: UploadFile = File(...),
    label: str = Form(...),
    key: str = Form(DEFAULT_KEY),
    threshold: float = Form(DEFAULT_WM_THRESHOLD),
):
    """
    水印确权接口：
    输入反演图 + 目标 label
    输出水印分数和是否检测到水印。
    """
    tmp_path = None

    try:
        tmp_path = save_upload_file(image)

        result = engine.detect_watermark(
            image_path=tmp_path,
            key_path=key,
            label=label,
            threshold=threshold,
        )

        debug_print(
            f"[/detect_watermark] image={image.filename or ''} label={label} "
            f"s_wm={float(result['s_wm']):.6f} threshold={float(result['threshold']):.6f} "
            f"detected={int(bool(result['detected']))} cos_clean={float(result['cos_clean']):.6f} "
            f"cos_wm={float(result['cos_wm']):.6f}"
        )

        return {
            "success": True,
            "label": str(result["label"]),
            "label_index": result["label_index"],
            "theta": result["theta"],
            "sin_theta": result["sin_theta"],
            "threshold": result["threshold"],
            "s_wm": result["s_wm"],
            "cos_clean": result["cos_clean"],
            "cos_wm": result["cos_wm"],
            "cos_wm_minus_clean": result["cos_wm_minus_clean"],
            "detected": bool(result["detected"]),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/both")
async def both(
    image: UploadFile = File(...),
    gallery_mode: str = Form("auto"),
    key: str = Form(DEFAULT_KEY),
    threshold: float = Form(DEFAULT_WM_THRESHOLD),
    label: Optional[str] = Form(None),
    topk: int = Form(5),
):
    """
    同时执行普通识别和水印检测。
    如果不传 label，就用识别结果的 pred_label 做水印检测。

    注意：
    /both 中的识别阶段实际使用 clean 还是 protected，由管理员统一 gallery_mode 设置决定。
    """
    tmp_path = None
    requested_gallery_mode = str(gallery_mode or "auto")

    try:
        tmp_path = save_upload_file(image)

        actual_gallery_mode = get_runtime_gallery_mode()
        gallery = get_gallery(actual_gallery_mode)

        debug_topk = max(int(topk), int(DEBUG_TOPK))
        pred = engine.predict(
            image_path=tmp_path,
            gallery_path=gallery["path"],
            topk=debug_topk,
        )

        debug_predict_result(
            endpoint="/both.predict",
            result=pred,
            gallery_mode=actual_gallery_mode,
            gallery_path=gallery["path"],
            image_name=image.filename or "",
        )

        detect_label = label
        if detect_label is None or detect_label == "":
            detect_label = str(pred["pred_label"])

        wm = engine.detect_watermark(
            image_path=tmp_path,
            key_path=key,
            label=detect_label,
            threshold=threshold,
        )

        debug_print(
            f"[/both.watermark] image={image.filename or ''} label={detect_label} "
            f"s_wm={float(wm['s_wm']):.6f} threshold={float(wm['threshold']):.6f} "
            f"detected={int(bool(wm['detected']))} cos_clean={float(wm['cos_clean']):.6f} "
            f"cos_wm={float(wm['cos_wm']):.6f}"
        )

        return {
            "success": True,
            "requested_gallery_mode": requested_gallery_mode,
            "gallery_mode": actual_gallery_mode,
            "runtime_config": get_runtime_config_public(),
            "predict": {
                "pred_label": str(pred["pred_label"]),
                "top1_cosine": pred["top1_cosine"],
                "topk": [
                    {
                        "label": str(item["label"]),
                        "cosine": item["cosine"],
                    }
                    for item in pred["topk"]
                ],
            },
            "watermark": {
                "label": str(wm["label"]),
                "label_index": wm["label_index"],
                "theta": wm["theta"],
                "sin_theta": wm["sin_theta"],
                "threshold": wm["threshold"],
                "s_wm": wm["s_wm"],
                "cos_clean": wm["cos_clean"],
                "cos_wm": wm["cos_wm"],
                "cos_wm_minus_clean": wm["cos_wm_minus_clean"],
                "detected": bool(wm["detected"]),
            },
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
            },
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==========================
# 日志查看接口
# ==========================

@app.get("/logs/enroll_face")
def get_enroll_logs():
    """
    查看管理员录入用户日志。
    """
    if not os.path.exists(ENROLL_LOG_CSV):
        return {
            "success": True,
            "logs": [],
        }

    rows = []
    with open(ENROLL_LOG_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return {
        "success": True,
        "num_logs": len(rows),
        "logs": rows[-100:],
    }


@app.get("/logs/delete_face")
def get_delete_logs():
    """
    查看管理员删除用户日志。
    """
    if not os.path.exists(DELETE_LOG_CSV):
        return {
            "success": True,
            "logs": [],
        }

    rows = []
    with open(DELETE_LOG_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return {
        "success": True,
        "num_logs": len(rows),
        "logs": rows[-100:],
    }


@app.get("/logs/score")
def get_score_logs():
    """
    查看模拟截获到的单图 score 日志。
    """
    if not os.path.exists(SCORE_LOG_CSV):
        return {
            "success": True,
            "logs": [],
        }

    rows = []
    with open(SCORE_LOG_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return {
        "success": True,
        "num_logs": len(rows),
        "logs": rows[-100:],
    }


@app.get("/logs/score_batch")
def get_score_batch_logs():
    """
    查看 /score_batch 批量打分日志。
    """
    if not os.path.exists(BATCH_SCORE_LOG_CSV):
        return {
            "success": True,
            "logs": [],
        }

    rows = []
    with open(BATCH_SCORE_LOG_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    return {
        "success": True,
        "num_logs": len(rows),
        "logs": rows[-100:],
    }