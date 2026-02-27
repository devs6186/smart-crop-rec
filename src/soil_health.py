"""
Soil health interpretation and suggestion messages.
- THRESHOLDS: approximate agronomic ranges (can be tuned from literature or local experts).
- get_soil_health_messages: returns list of warnings (e.g. low K, high pH).
- get_crop_specific_suggestions: for a recommended crop, returns actionable hints
  (e.g. banana + low K → suggest fertilizer) so the system answers "what should I do?"
"""

# Approximate agronomic ranges for interpretation (can be tuned from literature)
# N, P, K in kg/ha or relative units as in dataset; we use percentiles/ranges from EDA in practice.
THRESHOLDS = {
    "N": {"low": 40, "high": 90},
    "P": {"low": 25, "high": 55},
    "K": {"low": 25, "high": 45},
    "ph": {"low": 5.5, "high": 7.5},
    "rainfall": {"low": 80, "high": 220},
    "temperature": {"low": 18, "high": 32},
    "humidity": {"low": 50, "high": 85},
}

# Crop-specific hints: crop name -> list of (factor, message when factor is problematic)
CROP_SUGGESTIONS = {
    "banana": [
        ("K", "Banana is potassium-demanding. Low K may reduce yield; consider K fertilizer."),
        ("humidity", "Banana prefers high humidity for best growth."),
    ],
    "rice": [
        ("N", "Rice benefits from adequate nitrogen; low N can limit yield."),
        ("rainfall", "Rice typically needs sufficient water/rainfall."),
    ],
    "maize": [
        ("N", "Maize is nitrogen-responsive; consider N application if low."),
        ("temperature", "Maize prefers warm temperatures; very low temp can delay growth."),
    ],
    "cotton": [
        ("K", "Cotton yield and fibre quality respond to potassium."),
    ],
    "jute": [
        ("rainfall", "Jute requires ample moisture; low rainfall may affect fibre quality."),
    ],
    "coffee": [
        ("temperature", "Coffee prefers moderate temperatures; very high temp can stress plants."),
        ("ph", "Coffee often grows in slightly acidic soils; check pH suitability."),
    ],
    "default": [
        ("N", "Nitrogen influences vegetative growth; consider soil test and fertilizer if low."),
        ("P", "Phosphorus supports root and flowering; low P can limit yield."),
        ("K", "Potassium helps stress tolerance and quality; low K may reduce yield."),
    ],
}


def _get_level(value: float, key: str) -> str:
    """Return 'low', 'ok', or 'high' based on thresholds."""
    t = THRESHOLDS.get(key, {})
    if not t:
        return "ok"
    low, high = t.get("low", 0), t.get("high", 100)
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "ok"


def get_soil_health_messages(feature_dict: dict) -> list[str]:
    """
    feature_dict: keys like N, P, K, temperature, humidity, ph, rainfall (numeric values).
    Returns a list of short human-readable messages about soil/climate conditions.
    """
    messages = []
    for key, value in feature_dict.items():
        level = _get_level(value, key)
        if level == "low":
            if key == "N":
                messages.append("Low nitrogen detected. Consider nitrogen fertilizer for better vegetative growth.")
            elif key == "P":
                messages.append("Low phosphorus detected. Phosphorus supports root development and flowering.")
            elif key == "K":
                messages.append("Low potassium detected. Banana and other K-loving crops may yield poorly; consider fertilizer before planting.")
            elif key == "ph":
                messages.append("Soil pH is low (acidic). Some crops prefer neutral to slightly acidic pH; consider liming if needed.")
            elif key == "rainfall":
                messages.append("Low rainfall expected. Prefer drought-tolerant crops or plan for irrigation.")
            elif key == "temperature":
                messages.append("Low temperature. Cold-sensitive crops may be at risk; choose suitable varieties.")
            elif key == "humidity":
                messages.append("Low humidity. Irrigation or mulching can help in dry conditions.")
        elif level == "high":
            if key == "N":
                messages.append("High nitrogen. Good for leafy crops; avoid excess to prevent lodging.")
            elif key == "ph":
                messages.append("Soil pH is high (alkaline). Some crops prefer neutral to slightly acidic soils.")
            elif key == "rainfall":
                messages.append("High rainfall expected. Ensure drainage and disease management for susceptible crops.")
            elif key == "temperature":
                messages.append("High temperature. Heat-tolerant crops are preferable.")
    return messages


def get_crop_specific_suggestions(crop_name: str, feature_dict: dict) -> list[str]:
    """
    For a recommended crop, return suggestions based on current soil/climate.
    E.g. if crop is banana and K is low, return the banana-K message.
    """
    suggestions = []
    crop_lower = (crop_name or "").strip().lower()
    hints = CROP_SUGGESTIONS.get(crop_lower, CROP_SUGGESTIONS["default"])
    for factor, message in hints:
        value = feature_dict.get(factor)
        if value is None:
            continue
        level = _get_level(value, factor)
        if level == "low" or (factor == "ph" and level != "ok"):
            suggestions.append(message)
    return suggestions
