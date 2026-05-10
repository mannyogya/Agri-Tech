"""
Platform endpoints: decision support, local signals, procurement trust, safety,
market/water insights, case studies, and offline hints. Educational only.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from regional_supply_core import build_regional_supplies
from sqlalchemy import text
from sqlalchemy.engine import Engine

# --- shared geo helpers (mirror main.py logic loosely) ---


def _money_region(lat: float, lon: float) -> tuple[str, str, str]:
    if 6 <= lat <= 37 and 68 <= lon <= 98:
        return ("INR", "₹", "India")
    if 14 <= lat <= 32.7 and -118 <= lon <= -86:
        return ("MXN", "MX$", "Mexico")
    if -35 <= lat <= -22 and 16 <= lon <= 33:
        return ("ZAR", "R", "Southern Africa")
    if 36 <= lat <= 72 and -10 <= lon <= 40:
        return ("EUR", "€", "Europe")
    if 24 <= lat <= 50 and -125 <= lon <= -66:
        return ("USD", "$", "United States")
    return ("USD", "$", "Global")


def _district_key(lat: float, lon: float) -> str:
    return f"{round(lat * 20) / 20:.2f}_{round(lon * 20) / 20:.2f}"


def ensure_platform_tables(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS local_outcomes (
                    id SERIAL PRIMARY KEY,
                    district_key TEXT NOT NULL,
                    disease_key TEXT NOT NULL,
                    outcome_score INTEGER NOT NULL,
                    regimen_hint TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
                """
            )
        )


# --- Pydantic ---


class DecisionSupportBody(BaseModel):
    disease: str
    risk_level: str = "medium"
    confidence: float = Field(0.5, ge=0, le=1)
    language: str = "en"
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    growth_stage: str | None = None
    weather_pressure: str | None = None  # e.g. wet_week, dry_spike, calm


class OutcomeReportBody(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    disease_key: str
    outcome_score: int = Field(..., ge=1, le=5)
    regimen_hint: str | None = None


class ProcurementTrustBody(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    location_label: str = ""
    treatments: list[dict[str, Any]] = Field(default_factory=list)


class FieldLogBody(BaseModel):
    transcript: str
    language: str = "en"


class SafetyCheckBody(BaseModel):
    active_ingredients: list[str]
    crop: str
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


# --- builders ---


def _decision_pack(body: DecisionSupportBody) -> dict[str, Any]:
    risk = (body.risk_level or "medium").lower()
    conf = body.confidence
    wet = (body.weather_pressure or "").lower()
    stage = (body.growth_stage or "unspecified").lower()

    severity_label = (
        "high_field_pressure" if risk == "high" else ("watch" if risk == "low" else "moderate_pressure")
    )

    checklist: list[dict[str, Any]] = []
    for day, action in [
        (1, "Walk the field margin and check upper/lower leaf surfaces for spread."),
        (3, "Compare new photos to your baseline — note edge vs patchy patterns."),
        (7, "If symptoms worsen or spread >30% canopy, escalate to extension/agronomist."),
        (14, "Rotate active groups if you repeat fungicide/herbicide; record products used."),
    ]:
        checklist.append({"day_offset": day, "action": action})

    if "wet" in wet or "rain" in wet:
        checklist.insert(
            1,
            {
                "day_offset": 2,
                "action": "Wet weather extends infection — shorten scouting interval; avoid spraying in heavy rain.",
            },
        )

    resistance = [
        "Alternate FRAC/HRAC/IRAC groups across applications; never repeat the same single-site class back-to-back when label allows rotation.",
        "Use labeled mixtures only as directed; partial rates increase resistance selection pressure.",
    ]

    weather_note = "Weather pressure unspecified — match irrigation/fungicide timing to local forecasts."
    if "wet" in wet or "humid" in wet:
        weather_note = "High humidity/wet canopy favors many foliar pathogens — prioritize coverage and protectant timing."
    elif "dry" in wet or "drought" in wet:
        weather_note = "Dry stress can mimic or mask symptoms — confirm soil moisture before aggressive chemical moves."

    if "flower" in stage or "fruit" in stage:
        weather_note += " Reproductive stage: pay extra attention to PHI on every product you consider."

    phi_rei = (
        "PHI (pre-harvest interval) and REI (re-entry) are legal minimums on the registered label — "
        "they vary by crop, rate, and formulation. This app cannot compute PHI; read the label every time."
    )

    trust = {
        "confidence_band": "high" if conf >= 0.75 else ("medium" if conf >= 0.45 else "low"),
        "what_model_saw": "Image classifier confidence only — it does not see weather, soil tests, or nutrition.",
        "when_to_escalate": [
            "Confidence under 45%",
            "Cash crop at reproductive stage",
            "Rapid symptom expansion after a supposedly effective spray",
        ],
    }

    return {
        "severity_framing": severity_label,
        "growth_stage_context": stage,
        "monitoring_checklist": checklist,
        "resistance_management": resistance,
        "phi_rei_reminder": phi_rei,
        "weather_adjustment_note": weather_note,
        "trust": trust,
        "source": "api",
    }


def _structure_field_log(transcript: str, language: str) -> dict[str, Any]:
    t = transcript.strip()
    low = t.lower()
    log_type = "note"
    if any(k in low for k in ("spray", "fungicide", "pesticide", "herbicide", "insect")):
        log_type = "crop_protection_event"
    elif any(k in low for k in ("irrigation", "water", "flood", "dry")):
        log_type = "water_event"
    elif any(k in low for k in ("harvest", "cut", "pick")):
        log_type = "harvest_event"
    elif any(k in low for k in ("yellow", "spot", "rot", "wilting")):
        log_type = "symptom_observation"

    crops = []
    for name in (
        "tomato",
        "wheat",
        "rice",
        "maize",
        "corn",
        "potato",
        "cotton",
        "soy",
        "chili",
        "grape",
    ):
        if name in low:
            crops.append(name)

    products: list[str] = []
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s[a-z]+)?)\b", transcript):
        word = m.group(1)
        if len(word) > 3 and word.lower() not in ("the", "and", "this", "that", "when"):
            products.append(word)
    products = list(dict.fromkeys(products))[:5]

    return {
        "structured": {
            "type": log_type,
            "crops_guessed": crops,
            "mentioned_product_like_tokens": products,
            "raw_transcript": t,
            "language": language,
        },
        "suggested_title": f"{log_type.replace('_', ' ')} — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "next_steps": [
            "Attach a photo of the same bed/row within 24h for a dated timeline.",
            "If crop protection: note rate and tank volume when you confirm from label.",
        ],
        "source": "api",
    }


def _safety_flags(ingredients: list[str], crop: str, lat: float, lon: float) -> dict[str, Any]:
    _, _, region = _money_region(lat, lon)
    crop_l = crop.lower()
    flags: list[dict[str, str]] = []

    joined = " ".join(ingredients).lower()
    if any(x in joined for x in ("glyphosate", "paraquat", "2,4-d", "atrazine")):
        flags.append(
            {
                "level": "caution",
                "message": "Herbicide injury risk to sensitive crops via drift/volatilization — verify neighbor crops and wind.",
            }
        )
    if any(x in joined for x in ("chlorpyrifos", "methyl parathion")):
        flags.append(
            {
                "level": "strong_caution",
                "message": "Organophosphate handling requires PPE per label; many geographies restrict use.",
            }
        )
    if "tomato" in crop_l and "copper" in joined:
        flags.append(
            {
                "level": "info",
                "message": "Copper can russet fruit under slow drying — check label PHI and formulation notes.",
            }
        )

    flags.append(
        {
            "level": "info",
            "message": f"Region hint: {region}. Registrations differ — only use products approved for {crop} where you farm.",
        }
    )

    return {
        "flags": flags,
        "tank_mix_note": "Physical compatibility and crop safety are label-specific; perform jar tests only as label/extension allows.",
        "mrl_note": "Export crops may have stricter residue limits than domestic — confirm markets before spraying.",
        "source": "api",
    }


def _market_pulse(lat: float, lon: float, commodity: str) -> dict[str, Any]:
    h = int(hashlib.sha256(f"{lat:.2f}{lon:.2f}{commodity}".encode()).hexdigest()[:8], 16)
    currency, sym, region = _money_region(lat, lon)
    trend = ["firm", "soft", "mixed"][h % 3]
    seasonal_index = 40 + (h % 55)
    return {
        "region": region,
        "commodity": commodity or "general_inputs",
        "seasonal_demand_index": seasonal_index,
        "trend_word": trend,
        "narrative": (
            "Indicative only — built from coarse location hashing, not live exchange data. "
            "Use mandi/terminal prices for trading decisions."
        ),
        "source": "api",
    }


def _water_climate(lat: float, lon: float) -> dict[str, Any]:
    h = int(hashlib.sha256(f"{lat:.2f}{lon:.2f}".encode()).hexdigest()[:8], 16)
    # Very rough latitude bands
    drought_watch = "elevated" if abs(lat) < 35 and h % 5 == 0 else "normal"
    flood_watch = "elevated" if (h % 7 == 0) else "normal"

    et_hint = (
        "High solar angles near the equator — short, frequent irrigations may outperform deep rare flooding on light soils."
        if abs(lat) < 23
        else "Temperate/mid-latitude — check soil moisture before dosing; disease favorability changes with leaf wetness hours."
    )

    return {
        "drought_risk_band": drought_watch,
        "flood_excess_rain_band": flood_watch,
        "irrigation_hint": et_hint,
        "canopy_photo_tip": "Weekly phone photos of the same row build a crude stress timeline — not NDVI, but useful for audits.",
        "source": "api",
    }


def _case_studies(lat: float, lon: float, crop: str | None) -> dict[str, Any]:
    currency, _, region = _money_region(lat, lon)
    crop_f = (crop or "mixed").lower()
    studies = [
        {
            "id": "cs_01",
            "title": "Rotating FRAC groups during pepper fruit rot pressure",
            "region": "humid tropics (example)",
            "crop": "pepper",
            "timeline_days": 21,
            "intervention": "Alternated protectant + systemic per label; improved canopy airflow.",
            "outcome": "Slowed lesion expansion; yield saved vs untreated strip.",
            "anonymized": True,
        },
        {
            "id": "cs_02",
            "title": "Irrigation timing vs fungal splash (field tomato)",
            "region": region,
            "crop": "tomato",
            "timeline_days": 14,
            "intervention": "Shifted drip timing; avoided overhead during disease onset.",
            "outcome": "Reduced new infections on lower canopy in peer review plot.",
            "anonymized": True,
        },
        {
            "id": "cs_03",
            "title": "Herbicide drift complaint prevention",
            "region": "broadacre (example)",
            "crop": "soy",
            "timeline_days": 3,
            "intervention": "Wind <10 km/h, drift-reduction nozzle, buffer to sensitive field.",
            "outcome": "Zero off-target calls that season (farmer-reported).",
            "anonymized": True,
        },
    ]
    if "rice" in crop_f or "wheat" in crop_f:
        studies.insert(
            0,
            {
                "id": "cs_crop",
                "title": f"Nutrition correction before fungicide spend ({crop_f})",
                "region": region,
                "crop": crop_f,
                "timeline_days": 10,
                "intervention": "Soil test showed N imbalance; adjusted topdress before spraying.",
                "outcome": "Better response to fungicide program vs previous season.",
                "anonymized": True,
            },
        )

    return {"items": studies[:5], "currency_context": currency, "source": "api"}


def _trust_card(confidence: float | None, disease: str | None) -> dict[str, Any]:
    c = 0.6 if confidence is None else max(0.01, min(0.99, confidence))
    return {
        "confidence": c,
        "provenance": " ONNX crop-disease classifier where deployed; mock or fallback when model not loaded.",
        "human_escalation": [
            {"channel": "district_agriculture_office", "when": "Before off-label or tank-mix experiments on cash crops"},
            {"channel": "certified_agronomist", "when": "Rapid defoliation or uncertain pesticide choice"},
        ],
        "disclaimer": "Educational — not an emergency service. For pesticide exposure call your local poison/emergency line.",
        "source": "api",
    }


def _offline_manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "endpoints": [
            {"path": "/platform/offline-manifest", "method": "GET", "cache_ttl_seconds": 86400},
            {"path": "/platform/trust-card", "method": "GET", "cache_ttl_seconds": 604800},
            {"path": "/platform/case-studies", "method": "GET", "cache_ttl_seconds": 86400},
            {"path": "/platform/water-climate", "method": "GET", "cache_ttl_seconds": 43200},
            {"path": "/platform/market-pulse", "method": "GET", "cache_ttl_seconds": 21600},
            {"path": "/regional-supplies", "method": "POST", "cache_ttl_seconds": 7200},
        ],
        "low_data_tips": [
            "Prefer thumbnail photos for diagnosis when bandwidth is limited.",
            "Cache last weather bundle and last platform JSON in AsyncStorage.",
        ],
        "source": "api",
    }


def _procurement_trust_layer(
    lat: float, lon: float, location_label: str, treatments: list[dict[str, Any]]
) -> dict[str, Any]:
    base = build_regional_supplies(lat, lon, location_label, treatments)
    stores = []
    for s in base.get("stores", []):
        h = hash(json.dumps(s, sort_keys=True)) % 3
        tier = ["community_trusted", "authorized_channel", "verify_in_person"][h]
        stores.append(
            {
                **s,
                "trust": {
                    "tier": tier,
                    "verified_label": tier != "verify_in_person",
                    "batch_trace_hint": "Ask for invoice/lot for crop protection buys; keep photo for warranty disputes.",
                    "return_policy_hint": "Ag shops rarely accept opened chemicals — confirm before breaking seals.",
                    "delivery_sla_hint": "Rural delivery varies; call ahead for bulk bags and drums.",
                },
            }
        )
    return {
        **base,
        "stores": stores,
        "procurement_disclaimer": "Tiers are heuristic channel types, not audited seller scores. Build verified seller programs separately.",
        "source": "api",
    }


def build_platform_router(engine: Engine) -> APIRouter:
    router = APIRouter(prefix="/platform", tags=["platform"])
    ensure_platform_tables(engine)

    @router.post("/decision-support")
    def decision_support(body: DecisionSupportBody) -> dict[str, Any]:
        return _decision_pack(body)

    @router.get("/local-efficacy")
    def local_efficacy(
        lat: float = Query(..., ge=-90, le=90),
        lon: float = Query(..., ge=-180, le=180),
        disease_key: str = Query("general"),
    ) -> dict[str, Any]:
        dk = _district_key(lat, lon)
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                    SELECT outcome_score, COUNT(*) AS n
                    FROM local_outcomes
                    WHERE district_key = :dk AND disease_key = :d
                    AND created_at > NOW() - INTERVAL '180 days'
                    GROUP BY outcome_score
                    ORDER BY outcome_score
                    """
                ),
                {"dk": dk, "d": disease_key[:120]},
            )
            rows = [dict(r._mapping) for r in result]

        # mock prior when empty
        mock_signal = [
            {"regimen": "protectant rotation", "avg_score": 4.1, "n": 12},
            {"regimen": "single-site repeat", "avg_score": 2.3, "n": 8},
        ]
        return {
            "district_key": dk,
            "disease_key": disease_key,
            "aggregates": rows,
            "anonymous_mock_comparison": mock_signal if not rows else [],
            "privacy_note": "Only coarse district buckets and scores you submit are stored; no names or plot maps.",
            "source": "api",
        }

    @router.post("/outcome-report")
    def outcome_report(body: OutcomeReportBody) -> dict[str, Any]:
        dk = _district_key(body.lat, body.lon)
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO local_outcomes (district_key, disease_key, outcome_score, regimen_hint)
                    VALUES (:district_key, :disease_key, :outcome_score, :regimen_hint)
                    """
                ),
                {
                    "district_key": dk,
                    "disease_key": body.disease_key[:200],
                    "outcome_score": body.outcome_score,
                    "regimen_hint": (body.regimen_hint or "")[:500],
                },
            )
        return {"ok": True, "district_key": dk, "source": "api"}

    @router.post("/procurement-trust")
    def procurement_trust(body: ProcurementTrustBody) -> dict[str, Any]:
        if not body.treatments:
            return {
                "region_label": body.location_label or "",
                "price_estimates": [],
                "stores": [],
                "disclaimer": "",
                "source": "api",
            }
        return _procurement_trust_layer(body.lat, body.lon, body.location_label, body.treatments)

    @router.post("/field-log-structure")
    def field_log_structure(body: FieldLogBody) -> dict[str, Any]:
        if not body.transcript.strip():
            raise HTTPException(status_code=400, detail="transcript required")
        return _structure_field_log(body.transcript, body.language)

    @router.post("/safety-check")
    def safety_check(body: SafetyCheckBody) -> dict[str, Any]:
        if not body.active_ingredients:
            raise HTTPException(status_code=400, detail="active_ingredients required")
        return _safety_flags(body.active_ingredients, body.crop, body.lat, body.lon)

    @router.get("/market-pulse")
    def market_pulse(
        lat: float = Query(..., ge=-90, le=90),
        lon: float = Query(..., ge=-180, le=180),
        commodity: str = Query("inputs"),
    ) -> dict[str, Any]:
        return _market_pulse(lat, lon, commodity)

    @router.get("/water-climate")
    def water_climate(
        lat: float = Query(..., ge=-90, le=90),
        lon: float = Query(..., ge=-180, le=180),
    ) -> dict[str, Any]:
        return _water_climate(lat, lon)

    @router.get("/case-studies")
    def case_studies(
        lat: float = Query(20.0, ge=-90, le=90),
        lon: float = Query(77.0, ge=-180, le=180),
        crop: str | None = Query(None),
    ) -> dict[str, Any]:
        return _case_studies(lat, lon, crop)

    @router.get("/trust-card")
    def trust_card(
        confidence: float | None = Query(None),
        disease: str | None = Query(None),
    ) -> dict[str, Any]:
        return _trust_card(confidence, disease)

    @router.get("/offline-manifest")
    def offline_manifest() -> dict[str, Any]:
        return _offline_manifest()

    return router
