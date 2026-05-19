import os
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse

from app_cli import ArcFaceCLI


# ==========================
# 默认配置
# ==========================
DEFAULT_WEIGHTS = r"weights\fei_r50-train\model.pt"
DEFAULT_GALLERY = r"weights\fei_r50-train\fei_gallery.pt"
DEFAULT_WM_GALLERY = r"weights\fei_r50_protected\fei_gallery_wm_theta090.pt"
DEFAULT_KEY = r"weights\fei_r50_protected\watermark_key_theta090.pt"

DEFAULT_NETWORK = "r50"
DEFAULT_DEVICE = "cpu"
DEFAULT_THRESHOLD = 0.085


app = FastAPI(title="ArcFace Watermark Local API")


# 启动时加载一次模型，避免每次请求都重新加载
engine = ArcFaceCLI(
    weights=DEFAULT_WEIGHTS,
    network=DEFAULT_NETWORK,
    device=DEFAULT_DEVICE,
)


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


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEFAULT_DEVICE,
        "weights": DEFAULT_WEIGHTS,
        "default_gallery": DEFAULT_GALLERY,
        "default_key": DEFAULT_KEY,
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    gallery: str = Form(DEFAULT_GALLERY),
    topk: int = Form(5),
):
    tmp_path = None

    try:
        tmp_path = save_upload_file(image)

        result = engine.predict(
            image_path=tmp_path,
            gallery_path=gallery,
            topk=topk,
        )

        return {
            "success": True,
            "pred_label": str(result["pred_label"]),
            "top1_cosine": result["top1_cosine"],
            "topk": [
                {
                    "label": str(item["label"]),
                    "cosine": item["cosine"],
                }
                for item in result["topk"]
            ],
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
    threshold: float = Form(DEFAULT_THRESHOLD),
):
    tmp_path = None

    try:
        tmp_path = save_upload_file(image)

        result = engine.detect_watermark(
            image_path=tmp_path,
            key_path=key,
            label=label,
            threshold=threshold,
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
    gallery: str = Form(DEFAULT_GALLERY),
    key: str = Form(DEFAULT_KEY),
    threshold: float = Form(DEFAULT_THRESHOLD),
    label: Optional[str] = Form(None),
    topk: int = Form(5),
):
    tmp_path = None

    try:
        tmp_path = save_upload_file(image)

        pred = engine.predict(
            image_path=tmp_path,
            gallery_path=gallery,
            topk=topk,
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

        return {
            "success": True,
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