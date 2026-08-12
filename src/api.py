"""
api.py
------
FastAPI serving layer for DocuGuard. Exposes:

  POST /predict         -> {tampered: bool, probability: float}
  POST /predict/explain  -> same, plus a base64 Grad-CAM heatmap overlay PNG
                             showing WHERE the model thinks tampering is

  GET  /health           -> readiness check (also reports whether the
                             checkpoint has been loaded)

Run:
    uvicorn api:app --reload --port 8000
(from the src/ directory, or `python -m uvicorn src.api:app` from the repo root)
"""
import base64
import io
import os

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from model import DocuGuardFusionModel
from dataset import rgb_transform, ela_transform
from features import compute_ela
from gradcam import gradcam_for_ela_branch, overlay_heatmap

CKPT_PATH = os.environ.get("DOCUGUARD_CKPT", "models/docuguard_best.pt")

app = FastAPI(title="DocuGuard", description="Synthetic ID-document forgery detection API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None


def get_model():
    global _model
    if _model is None:
        m = DocuGuardFusionModel()
        if os.path.exists(CKPT_PATH):
            m.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
        else:
            raise RuntimeError(f"Checkpoint not found at {CKPT_PATH}. Train the model first (see README).")
        m.eval()
        _model = m
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint_found": os.path.exists(CKPT_PATH)}


def _load_and_prep(file_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")
    rgb_t = rgb_transform(img).unsqueeze(0)
    ela_img = compute_ela(img, quality=90)
    ela_t = ela_transform(ela_img).unsqueeze(0)
    return img, rgb_t, ela_t


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = get_model()
    img, rgb_t, ela_t = _load_and_prep(await file.read())
    with torch.no_grad():
        logits = model(rgb_t, ela_t)
        prob = torch.sigmoid(logits).item()
    return JSONResponse({
        "tampered": bool(prob >= 0.5),
        "probability": round(prob, 4),
        "filename": file.filename,
    })


@app.post("/predict/explain")
async def predict_explain(file: UploadFile = File(...)):
    model = get_model()
    img, rgb_t, ela_t = _load_and_prep(await file.read())

    cam, prob = gradcam_for_ela_branch(model, rgb_t, ela_t)

    buf = io.BytesIO()
    overlay_heatmap(img, cam, "/tmp/_docuguard_overlay.png")
    with open("/tmp/_docuguard_overlay.png", "rb") as f:
        overlay_b64 = base64.b64encode(f.read()).decode("utf-8")

    return JSONResponse({
        "tampered": bool(prob >= 0.5),
        "probability": round(prob, 4),
        "filename": file.filename,
        "heatmap_overlay_png_base64": overlay_b64,
    })
