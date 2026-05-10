import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from PIL import Image
from sqlalchemy import create_engine, text

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore

app = FastAPI(title="Agritech")

_REPO_ROOT = Path(__file__).resolve().parent

# Lazy-loaded ONNX classifier (optional). Train locally: see ml/README.md
_ORT_SESSION: Any = None
_CLASS_NAMES: list[str] | None = None
_ADVICE_MAP: dict[str, Any] | None = None
# Set when files look valid but ONNX fails to load (shows in /model-status for debugging).
_LAST_LOAD_ERROR: str | None = None


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_model_loaded() -> bool:
    """Return True if ONNX session + labels + advice are ready."""
    global _ORT_SESSION, _CLASS_NAMES, _ADVICE_MAP, _LAST_LOAD_ERROR
    if ort is None:
        _LAST_LOAD_ERROR = "onnxruntime import failed (not installed)"
        return False
    if _ORT_SESSION is not None and _CLASS_NAMES is not None and _ADVICE_MAP is not None:
        return True

    model_path = (os.getenv("MODEL_PATH") or "").strip()
    if not model_path:
        cand = _REPO_ROOT / "plant_disease.onnx"
        model_path = str(cand) if cand.is_file() else ""

    if not model_path or not Path(model_path).is_file():
        _LAST_LOAD_ERROR = f"model file missing or not found: {model_path!r}"
        return False

    labels_path = (os.getenv("LABELS_PATH") or "").strip() or str(_REPO_ROOT / "labels.json")
    advice_path = (os.getenv("ADVICE_PATH") or "").strip() or str(_REPO_ROOT / "advice_by_class.json")

    labels_data = _load_json(Path(labels_path))
    if not labels_data or "classes" not in labels_data:
        _LAST_LOAD_ERROR = f"labels.json missing or has no 'classes' key: {labels_path!r}"
        return False

    advice = _load_json(Path(advice_path))
    if not isinstance(advice, dict):
        _LAST_LOAD_ERROR = f"advice JSON missing or not a dict: {advice_path!r}"
        return False

    try:
        _ORT_SESSION = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
    except Exception as e:
        _LAST_LOAD_ERROR = f"ONNX load failed: {e!r}"
        _ORT_SESSION = None
        _CLASS_NAMES = None
        _ADVICE_MAP = None
        return False

    _CLASS_NAMES = list(labels_data["classes"])
    _ADVICE_MAP = advice
    _LAST_LOAD_ERROR = None
    return True


def _preprocess_image(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize((224, 224), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    return np.expand_dims(arr, axis=0)


def _predict_class(raw: bytes) -> tuple[str, float] | None:
    if not _ensure_model_loaded():
        return None
    assert _ORT_SESSION is not None and _CLASS_NAMES is not None

    inp = _preprocess_image(raw)
    logits = _ORT_SESSION.run(None, {"input": inp})[0]
    logits = np.asarray(logits).reshape(-1)
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return _CLASS_NAMES[idx], conf


def _payload_from_model(class_key: str, confidence: float, language: str) -> dict[str, Any]:
    conf = round(min(0.99, max(0.01, confidence)), 4)
    if _ADVICE_MAP is not None:
        per_class = _ADVICE_MAP.get(class_key)
        if per_class:
            block = per_class.get(language) or per_class.get("en")
            if block:
                return {
                    "disease": block["disease"],
                    "confidence": conf,
                    "risk_level": block["risk_level"],
                    "advice": block["advice"],
                }
    pretty = class_key.replace("___", " — ").replace("_", " ")
    return {
        "disease": pretty,
        "confidence": conf,
        "risk_level": "medium",
        "advice": "No localized advice for this class yet — confirm with an agronomist. Keep improving training data and advice_by_class.json.",
    }


def _mock_payload(language: str) -> dict[str, Any]:
    responses = {
        "en": {
            "disease": "Early Blight",
            "confidence": 0.78,
            "risk_level": "medium",
            "advice": "Avoid overwatering and remove infected leaves.",
        },
        "hi": {
            "disease": "अर्ली ब्लाइट",
            "confidence": 0.78,
            "risk_level": "medium",
            "advice": "ज्यादा पानी न दें और संक्रमित पत्तियां हटा दें।",
        },
        "es": {
            "disease": "Tizón temprano",
            "confidence": 0.78,
            "risk_level": "medium",
            "advice": "Evita el exceso de riego y retira las hojas infectadas.",
        },
    }
    return responses.get(language, responses["en"])

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)


def save_diagnosis_row(
    *,
    client_id: str,
    language: str,
    symptom_text: str | None,
    disease: str,
    confidence: float,
    risk_level: str,
    advice: str,
    image_url: str | None = None,
):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO diagnoses
                  (client_id, language, symptom_text, disease, confidence, risk_level, advice, image_url)
                VALUES
                  (:client_id, :language, :symptom_text, :disease, :confidence, :risk_level, :advice, :image_url)
                """
            ),
            {
                "client_id": client_id,
                "language": language,
                "symptom_text": symptom_text,
                "disease": disease,
                "confidence": confidence,
                "risk_level": risk_level,
                "advice": advice,
                "image_url": image_url,
            },
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    ts = out.get("created_at")
    if isinstance(ts, datetime):
        out["created_at"] = ts.isoformat()
    return out


@app.get("/diagnoses")
def list_diagnoses(x_client_id: str | None = Header(default=None, alias="X-Client-Id")):
    client_id = x_client_id or "anonymous"

    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT id, created_at, language, symptom_text, disease, confidence, risk_level, advice
                FROM diagnoses
                WHERE client_id = :cid
                ORDER BY created_at DESC
                LIMIT 50
                """
            ),
            {"cid": client_id},
        )
        rows = [dict(r._mapping) for r in result]

    return {"items": [_serialize_row(r) for r in rows]}


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str = Form("en"),
):
    """Upload short audio; returns transcript text (OpenAI Whisper)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    buf = io.BytesIO(content)
    buf.name = audio.filename or "audio.m4a"

    lang_map = {"en": "en", "hi": "hi", "es": "es"}
    whisper_lang = lang_map.get(language, "en")

    client = OpenAI(api_key=api_key)
    tr = client.audio.transcriptions.create(
        model="whisper-1",
        file=buf,
        language=whisper_lang,
    )

    return {"text": tr.text}


@app.get("/")
def home():
    return {"message": "Welcome to the Agritech API!"}


@app.get("/model-status")
def model_status():
    """Whether ONNX classifier + labels + advice loaded (real predictions vs mock)."""
    ok = _ensure_model_loaded()
    out: dict[str, Any] = {
        "model_loaded": ok,
        "num_classes": len(_CLASS_NAMES) if _CLASS_NAMES else 0,
        "onnxruntime_installed": ort is not None,
    }
    if not ok and _LAST_LOAD_ERROR:
        out["load_error"] = _LAST_LOAD_ERROR
    return out


@app.post("/diagnose")
async def diagnose(
    image: UploadFile = File(...),
    language: str = Form("en"),
    symptom_text: str | None = Form(None),
    x_client_id: str | None = Header(default=None, alias="X-Client-Id"),
):
    client_id = x_client_id or "anonymous"

    raw = await image.read()
    payload: dict[str, Any]

    try:
        pred = _predict_class(raw)
    except Exception:
        pred = None

    if pred is not None:
        class_key, conf = pred
        payload = _payload_from_model(class_key, conf, language)
    else:
        payload = _mock_payload(language)

    save_diagnosis_row(
        client_id=client_id,
        language=language,
        symptom_text=symptom_text,
        disease=payload["disease"],
        confidence=payload["confidence"],
        risk_level=payload["risk_level"],
        advice=payload["advice"],
        image_url=image.filename,
    )

    return payload
