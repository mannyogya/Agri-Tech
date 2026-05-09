from fastapi import FastAPI, UploadFile, File, Form, Header
from fastapi.middleware.cors import CORSMiddleware
import os
from sqlalchemy import create_engine, text
from typing import Any
from datetime import datetime

app = FastAPI(title="Agritech")

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

@app.get("/")
def home():
    return {"message": "Welcome to the Agritech API!"}


@app.post("/diagnose")
async def diagnose(
    image: UploadFile = File(...),
    language: str = Form("en"),
    symptom_text: str | None = Form(None),
    x_client_id: str | None = Header(default=None, alias="X-Client-Id"),
):
    client_id = x_client_id or "anonymous"

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

    payload = responses.get(language, responses["en"])

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
