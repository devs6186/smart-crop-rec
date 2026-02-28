"""
Zone-based soil/climate defaults for crop recommendation.
State + district → N, P, K, temperature, humidity, ph, rainfall.
Used by app and tests; no Streamlit dependency.
"""

from src.config import (
    FEATURE_COLUMNS,
    ZONE_DEFAULTS,
    STATE_ZONE,
)


def _state_offset(state: str, district: str | None, feature: str) -> float:
    """Deterministic offset per state+district so recommendations vary meaningfully by region."""
    h = hash((state, district or "", feature)) % 100
    return (h - 50) / 50.0  # -1.0 to +1.0


def get_default_soil_climate(state: str | None, district: str | None = None) -> dict[str, float]:
    """
    State+district-specific soil/climate so ML recommendations vary by region.
    Returns N, P, K, temperature, humidity, ph, rainfall.
    """
    if not state or state not in STATE_ZONE:
        return {
            "N": 50.0, "P": 50.0, "K": 50.0,
            "temperature": 25.0, "humidity": 65.0, "ph": 6.5, "rainfall": 120.0,
        }
    zone = STATE_ZONE[state]
    base = ZONE_DEFAULTS[zone]
    out = {}
    for k in FEATURE_COLUMNS:
        v = base[k]
        delta = _state_offset(state, district, k)
        if k in ("temperature", "humidity", "ph"):
            v = max(15.0, min(45.0 if k == "temperature" else 100.0 if k == "humidity" else 9.5, v + delta * 8))
        elif k == "rainfall":
            v = max(20.0, min(300.0, v + delta * 55))
        else:
            v = max(0, min(205 if k == "K" else 145 if k == "P" else 140, v + int(delta * 22)))
        out[k] = round(v, 2) if isinstance(v, float) else v
    return out
