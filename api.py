"""
FastAPI REST API for Chest X-Ray Disease Detection.
Endpoints:
  GET  /          → health check
  POST /predict   → predict class + return Grad-CAM heatmap
  GET  /metrics   → return latest evaluation metrics
"""
import io
import os
import base64
import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch

from src.config import CHECKPOINT, OUTPUT_DIR, CLASS_NAMES, get_device
from src.model import build_model, load_checkpoint
from src.gradcam import predict_with_gradcam, pil_to_bytes

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Chest X-Ray Disease Detection API",
    description="Binary classification: NORMAL vs PNEUMONIA using ResNet-18 + Grad-CAM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global model (loaded once at startup) ────────────────────────────────────
device = get_device()
model  = None


@app.on_event("startup")
async def startup_event():
    global model
    logger.info(f"Loading model from {CHECKPOINT}")
    m = build_model(pretrained=False)
    if os.path.exists(CHECKPOINT):
        model = load_checkpoint(m, CHECKPOINT, device)
        logger.info("Model checkpoint loaded.")
    else:
        logger.warning("No checkpoint found – using untrained model. Run train.py first.")
        model = m.to(device).eval()


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "model": "ResNet-18", "classes": CLASS_NAMES}


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Upload a chest X-ray image (JPEG/PNG) and receive:
    - Predicted class (NORMAL / PNEUMONIA)
    - Confidence score
    - Grad-CAM heatmap (Base64 PNG)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot read image: {e}")

    result = predict_with_gradcam(model, image, device)

    # Encode Grad-CAM image to base64
    gradcam_bytes  = pil_to_bytes(result["gradcam_image"])
    gradcam_b64    = base64.b64encode(gradcam_bytes).decode()
    original_bytes = pil_to_bytes(image.resize((224, 224)))
    original_b64   = base64.b64encode(original_bytes).decode()

    return JSONResponse({
        "class_name":    result["class_name"],
        "class_idx":     result["class_idx"],
        "confidence":    result["confidence"],
        "probabilities": result["probabilities"],
        "gradcam_image": gradcam_b64,   # base64 PNG
        "original_image": original_b64, # base64 PNG
    })


@app.get("/metrics", tags=["Evaluation"])
def get_metrics():
    """Return the latest evaluation metrics from outputs/metrics.json."""
    path = os.path.join(OUTPUT_DIR, "metrics.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Metrics not yet generated. Run evaluate first.")
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False)
