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

# Shown with chemical suggestions — not a substitute for the registered product label.
DEFAULT_TREATMENT_DISCLAIMER: dict[str, str] = {
    "en": "Products, rates, and pre-harvest intervals vary by country and registration. Follow the printed product label and your local agricultural extension. This is educational information only.",
    "hi": "दवाएं और खुराक देश और रजिस्ट्रेशन के अनुसार अलग होती हैं। उत्पाद लेबल और स्थानीय कृषि सलाह मानें — यह शैक्षिक जानकारी है।",
    "es": "Los productos, dosis y carencias de cosecha varían por país y registro. Siga la etiqueta del producto y su extensión agrícola local. Esto es solo información educativa.",
}


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
                raw_t = block.get("treatments")
                treatments = raw_t if isinstance(raw_t, list) else []
                disc = block.get("treatment_disclaimer") or DEFAULT_TREATMENT_DISCLAIMER.get(
                    language
                ) or DEFAULT_TREATMENT_DISCLAIMER["en"]
                return {
                    "disease": block["disease"],
                    "confidence": conf,
                    "risk_level": block["risk_level"],
                    "advice": block["advice"],
                    "treatments": treatments,
                    "treatment_disclaimer": disc,
                }
    pretty = class_key.replace("___", " — ").replace("_", " ")
    return {
        "disease": pretty,
        "confidence": conf,
        "risk_level": "medium",
        "advice": "No localized advice for this class yet — confirm with an agronomist. Keep improving training data and advice_by_class.json.",
        "treatments": [],
        "treatment_disclaimer": DEFAULT_TREATMENT_DISCLAIMER.get(language)
        or DEFAULT_TREATMENT_DISCLAIMER["en"],
    }


def _mock_payload(language: str) -> dict[str, Any]:
    responses: dict[str, dict[str, Any]] = {
        "en": {
            "disease": "Early Blight",
            "confidence": 0.78,
            "risk_level": "medium",
            "advice": "Avoid overwatering and remove infected leaves.",
            "treatments": [
                {
                    "name": "Chlorothalonil-based fungicide (e.g. Bravo / equivalent)",
                    "active_ingredient": "Chlorothalonil",
                    "when_to_apply": "At first symptoms; repeat on the label schedule (often 7–14 days) while wet weather continues.",
                    "rate": "Use only the rate on your product label for tomatoes (foliar). Do not exceed the label maximum per season.",
                    "notes": "Observe PHI and REI on the label. Alternate FRAC groups with other modes of action.",
                },
                {
                    "name": "Azoxystrobin + difenoconazole or similar premix (check local registration)",
                    "active_ingredient": "Mixed (strobilurin + DMI)",
                    "when_to_apply": "Use according to label for early blight; often rotated with chlorothalonil or mancozeb programs.",
                    "rate": "Label-only; premix ratios are fixed—do not exceed labeled applications per year.",
                    "notes": "Resistance management: rotate with non-strobilurin partners.",
                },
            ],
            "treatment_disclaimer": DEFAULT_TREATMENT_DISCLAIMER["en"],
        },
        "hi": {
            "disease": "अर्ली ब्लाइट",
            "confidence": 0.78,
            "risk_level": "medium",
            "advice": "ज्यादा पानी न दें और संक्रमित पत्तियां हटा दें।",
            "treatments": [
                {
                    "name": "क्लोरोथालोनिल आधारित फ़फूंदनाशी (उदाहरण: ब्रावो जैसे)",
                    "active_ingredient": "Chlorothalonil",
                    "when_to_apply": "पहले लक्षणों पर; गीले मौसम में लेबल अनुसार दोहराएं।",
                    "rate": "केवल उत्पाद लेबल पर टमाटर के लिए दिखाई दर का उपयोग करें।",
                    "notes": "कटाई से पहले अंतराल (PHI) और पुनः प्रवेश (REI) लेबल से मानें।",
                }
            ],
            "treatment_disclaimer": DEFAULT_TREATMENT_DISCLAIMER["hi"],
        },
        "es": {
            "disease": "Tizón temprano",
            "confidence": 0.78,
            "risk_level": "medium",
            "advice": "Evita el exceso de riego y retira las hojas infectadas.",
            "treatments": [
                {
                    "name": "Fungicida a base de clorotalonil (ej. Bravo u homólogo)",
                    "active_ingredient": "Clorotalonil",
                    "when_to_apply": "Al primer síntoma; repetir según la etiqueta mientras haya presión de enfermedad.",
                    "rate": "Solo la dosis autorizada en la etiqueta para tomate (foliar).",
                    "notes": "Respeta IPA y tiempo de reingreso indicados en la etiqueta.",
                }
            ],
            "treatment_disclaimer": DEFAULT_TREATMENT_DISCLAIMER["es"],
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
    treatments_json: str | None = None,
):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO diagnoses
                  (client_id, language, symptom_text, disease, confidence, risk_level, advice, image_url, treatments_json)
                VALUES
                  (:client_id, :language, :symptom_text, :disease, :confidence, :risk_level, :advice, :image_url, :treatments_json)
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
                "treatments_json": treatments_json,
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
    tj = out.pop("treatments_json", None)
    if tj:
        try:
            parsed = json.loads(tj)
            out["treatments"] = parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            out["treatments"] = []
    else:
        out["treatments"] = []
    return out


@app.get("/diagnoses")
def list_diagnoses(x_client_id: str | None = Header(default=None, alias="X-Client-Id")):
    client_id = x_client_id or "anonymous"

    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT id, created_at, language, symptom_text, disease, confidence, risk_level, advice, treatments_json
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

    treatments = payload.get("treatments")
    t_json = json.dumps(treatments) if isinstance(treatments, list) else None

    save_diagnosis_row(
        client_id=client_id,
        language=language,
        symptom_text=symptom_text,
        disease=payload["disease"],
        confidence=payload["confidence"],
        risk_level=payload["risk_level"],
        advice=payload["advice"],
        image_url=image.filename,
        treatments_json=t_json,
    )

    return payload
