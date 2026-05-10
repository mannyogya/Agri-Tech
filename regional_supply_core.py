"""Shared regional price + store channel helpers for supplies APIs."""

from __future__ import annotations

from typing import Any


def js_string_hash(s: str) -> int:
    """32-bit style string hash (stable across processes, similar to TS imul loop)."""
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    if h & 0x80000000:
        h = -((h ^ 0xFFFFFFFF) + 1)
    return abs(h)


def infer_money_region(lat: float, lon: float) -> tuple[str, str, str]:
    if 6 <= lat <= 37 and 68 <= lon <= 98:
        return ("INR", "\u20b9", "India")
    if 14 <= lat <= 32.7 and -118 <= lon <= -86:
        return ("MXN", "MX$", "Mexico")
    if -35 <= lat <= -22 and 16 <= lon <= 33:
        return ("ZAR", "R", "Southern Africa")
    if 36 <= lat <= 72 and -10 <= lon <= 40:
        return ("EUR", "\u20ac", "Europe")
    if 24 <= lat <= 50 and -125 <= lon <= -66:
        return ("USD", "$", "United States")
    return ("USD", "$", "your region")


def mk_price_estimate(name: str, currency: str, symbol: str) -> dict[str, Any]:
    seed = js_string_hash(name + currency)
    is_inr = currency == "INR"
    if is_inr:
        low = 180 + (seed % 4200)
        high = low + 120 + (seed % 2800)
    else:
        low = int(8 + (seed % 220) + (seed % 17) * 3)
        high = int(low + 6 + (seed % 140))
    high = max(high, low + 1)
    unit_hint = "usual farm-shop pack"
    nlow = name.lower()
    if any(x in nlow for x in ("spray", "liquid", "liter", " l")):
        unit_hint = "per bottle or per liter (shop size varies)"
    if any(
        x in nlow for x in ("kg", "powder", "granule", "fertilizer", "urea", "manure")
    ):
        unit_hint = "per kg or bag size shown in store"
    return {
        "product_name": name,
        "price_low": float(low),
        "price_high": float(high),
        "currency": currency,
        "currency_symbol": symbol,
        "unit_hint": unit_hint,
    }


def mk_stores(lat: float, lon: float, region_label: str) -> list[dict[str, Any]]:
    seed = js_string_hash(f"{lat:.2f}|{lon:.2f}")
    if infer_money_region(lat, lon)[0] == "INR":
        base = [
            ("Krishi Kendra / input shop", "Near main market area"),
            ("Cooperative society store", "District agri network"),
            ("Private agro dealer", "Ask for crop medicine section"),
        ]
    else:
        base = [
            ("Farm supply & co-op", "Town or highway ag cluster"),
            ("Authorized ag dealer", "Brand sign outside shop"),
            ("Feed & farm store", "May carry crop protection"),
        ]
    out: list[dict[str, Any]] = []
    for i, (name, hint) in enumerate(base):
        dist = round(4 + ((seed + i * 79) % 34) + (i * 3))
        out.append(
            {
                "name": name,
                "distance_km": float(dist),
                "address_hint": hint,
            }
        )
    return out


def build_regional_supplies(
    lat: float,
    lon: float,
    location_label: str,
    treatments: list[dict[str, Any]],
) -> dict[str, Any]:
    currency, symbol, region_hint = infer_money_region(lat, lon)
    region_label = (
        location_label.strip() if location_label and location_label.strip() else region_hint
    )
    names = [t.get("name") for t in treatments if t.get("name")]
    price_estimates = [mk_price_estimate(n, currency, symbol) for n in names]
    stores = mk_stores(lat, lon, region_label)
    disclaimer = (
        "Indicative ranges only — real prices change by brand, size, tax, and season. "
        "Confirm in store. Store list is typical channel types near your coordinates, not live inventory."
    )
    return {
        "region_label": region_label,
        "price_estimates": price_estimates,
        "stores": stores,
        "disclaimer": disclaimer,
        "source": "api",
    }
