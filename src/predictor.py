"""
Prediction module: load trained artifacts and expose predict_crop().

Returns top-5 crops ranked by profit (or balanced score), each with:
  - suitability confidence (ML signal)
  - economic analysis (yield, revenue, cost, profit)
  - risk assessment (climate + disease)
  - explanation text
  - soil health messages

Backward-compatible: land_size_bigha, state, district are optional.
If omitted the function behaves like the original top-3 suitability-only mode,
but using national averages and returning top-5 instead of 3.
"""

import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    MODELS_DIR,
    FEATURE_COLUMNS,
    MODEL_ARTIFACT_NAME,
    SCALER_ARTIFACT_NAME,
    ENCODER_ARTIFACT_NAME,
    METADATA_FNAME,
    TOP_K_CROPS,
    CANDIDATES_POOL,
    MIN_SUITABILITY_PCT,
    SCORING_MODE,
    W_SUITABILITY,
    W_PROFIT,
    W_RISK,
    CROP_MIN_LAND_ACRES,
    DEFAULT_MIN_LAND_ACRES,
)
from src.soil_health import get_soil_health_messages, get_crop_specific_suggestions
from src.explainer import (
    get_importance_dict,
    explain_prediction_with_importance,
    explain_prediction_shap_text,
)
from src.region_data_loader import get_region_context, bigha_to_acres, get_bigha_factor
from src.profit_engine import compute_profit, normalise_profit_scores, rank_by_profit
from src.risk_engine import (
    get_disease_risks,
    compute_composite_risk,
    get_risk_label,
    normalise_risk_scores,
    get_all_prevention_measures,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_artifacts(models_dir: Path | None = None):
    """Load model, scaler, label encoder, and metadata from models/."""
    d = models_dir or MODELS_DIR
    model         = joblib.load(d / MODEL_ARTIFACT_NAME)
    scaler        = joblib.load(d / SCALER_ARTIFACT_NAME)
    label_encoder = joblib.load(d / ENCODER_ARTIFACT_NAME)
    with open(d / METADATA_FNAME) as f:
        metadata = json.load(f)
    return model, scaler, label_encoder, metadata


def _feature_dict(N, P, K, temperature, humidity, ph, rainfall) -> dict:
    """Build ordered dict of inputs for scaling and explanation."""
    return {
        "N":           float(N),
        "P":           float(P),
        "K":           float(K),
        "temperature": float(temperature),
        "humidity":    float(humidity),
        "ph":          float(ph),
        "rainfall":    float(rainfall),
    }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _balanced_score(
    suitability_norm: float,
    profit_norm: float,
    risk_norm: float,
) -> float:
    """
    Weighted score for balanced ranking mode.
    Higher is better; risk_norm is subtracted (higher risk → lower score).
    """
    return round(
        W_SUITABILITY * suitability_norm
        + W_PROFIT     * profit_norm
        - W_RISK       * risk_norm,
        4,
    )


def _normalise_list(values: list[float]) -> list[float]:
    """Min-max normalise a list to 0-1. Returns all zeros if span is zero."""
    mn, mx = min(values), max(values)
    span = mx - mn
    if span == 0:
        return [0.5] * len(values)
    return [(v - mn) / span for v in values]


# ---------------------------------------------------------------------------
# Main prediction API
# ---------------------------------------------------------------------------

def predict_crop(
    N, P, K, temperature, humidity, ph, rainfall,
    land_size_bigha: float = 1.0,
    state: str | None = None,
    district: str | None = None,
    scoring_mode: str | None = None,
    models_dir: Path | None = None,
    X_test_sample=None,
    y_test_sample=None,
) -> dict:
    """
    Full prediction + economic analysis API.

    Parameters
    ----------
    N, P, K, temperature, humidity, ph, rainfall : numeric
        Soil and climate inputs.
    land_size_bigha : float
        Farm size in bigha. Converted to acres using state-specific factor.
    state : str or None
        Indian state for region-specific data lookup.
    district : str or None
        District within the state (preferred over state-level data).
    scoring_mode : str or None
        "profit" (default) or "balanced". Overrides config default if given.
    models_dir : Path or None
        Override for models directory.
    X_test_sample, y_test_sample : optional arrays
        Used only for permutation importance if no importance dict exists.

    Returns
    -------
    dict with keys:
        top5              : list of crop result dicts (see below)
        explanation       : str (global feature-based explanation)
        soil_health_messages : list[str]
        crop_suggestions  : list[str] (for top recommended crop)
        land_size_acres   : float
        bigha_factor      : float (acres per bigha for this state)
        scoring_mode      : str
        region            : dict {state, district}

    Each crop result dict:
        rank, crop, suitability_pct,
        yield_q_per_bigha, total_production_quintals,
        price_per_quintal, gross_revenue_inr, input_cost_inr,
        net_profit_inr, profit_per_bigha_inr, roi_pct,
        risk_score, risk_label,
        disease_risks (list), prevention_measures (list),
        explanation (crop-specific),
        data_confidence
    """
    mode = (scoring_mode or SCORING_MODE).lower()
    model, scaler, label_encoder, metadata = load_artifacts(models_dir)
    feature_names = metadata.get("feature_names", FEATURE_COLUMNS)

    # Build feature vector (DataFrame preserves feature names for scaler)
    fd = _feature_dict(N, P, K, temperature, humidity, ph, rainfall)
    X  = pd.DataFrame([[fd[c] for c in feature_names]], columns=feature_names)
    X_scaled = scaler.transform(X)

    # Probabilities for all classes
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
    else:
        pred  = model.predict(X_scaled)[0]
        probs = np.zeros(len(label_encoder.classes_))
        probs[pred] = 1.0

    classes    = label_encoder.classes_.tolist()
    idx_sorted = np.argsort(probs)[::-1]

    # ------------------------------------------------------------------
    # Candidate selection: suitability gate → profit ranking
    #
    # Rule: only consider crops whose ML confidence is >= MIN_SUITABILITY_PCT.
    # This prevents 0%-suitable fruits from dominating purely on profit.
    #
    # Fallback: if fewer than 3 crops pass the threshold (e.g. small/noisy
    # training set), lower the threshold progressively until we have at
    # least 3 candidates — ensuring the app always shows something useful.
    # ------------------------------------------------------------------
    min_prob = MIN_SUITABILITY_PCT / 100.0

    above = [i for i in idx_sorted if probs[i] >= min_prob]
    relaxed_threshold = min_prob  # track what threshold we ended up using

    # Progressive threshold relaxation so we always have ≥ TOP_K_CROPS candidates
    if len(above) < TOP_K_CROPS:
        for relaxed in [0.02, 0.01, 0.005, 0.0]:
            above = [i for i in idx_sorted if probs[i] >= relaxed]
            if len(above) >= TOP_K_CROPS:
                relaxed_threshold = relaxed
                break

    # Cap pool at CANDIDATES_POOL before profit-ranking
    top_indices = above[:CANDIDATES_POOL]
    # Track which crops genuinely passed the original gate (>= MIN_SUITABILITY_PCT)
    genuine_indices = set(i for i in top_indices if probs[i] >= min_prob)

    # Land size in acres
    bigha_factor    = get_bigha_factor(state)
    land_size_acres = bigha_to_acres(land_size_bigha, state)

    # ---------------------------------------------------------------------------
    # Build per-crop data: suitability + profit + risk
    # ---------------------------------------------------------------------------
    crop_data = []
    for idx in top_indices:
        crop = classes[idx]
        conf = float(probs[idx])

        # Region context (yield, price, cost, vulnerability)
        region_ctx = get_region_context(crop, state, district)

        # Profit computation
        profit_data = compute_profit(crop, region_ctx, land_size_acres, conf)

        # Risk computation
        diseases     = get_disease_risks(crop)
        climate_vuln = region_ctx.get("vulnerability_index", 50.0)
        risk_score   = compute_composite_risk(climate_vuln, diseases)
        risk_label   = get_risk_label(risk_score)
        prevention   = get_all_prevention_measures(diseases)

        # Per-crop explanation (soil suggestion for this crop)
        crop_suggestion = get_crop_specific_suggestions(crop, fd)

        # Convert per-acre profit metrics to per-bigha for display
        yield_q_per_bigha    = round(profit_data["effective_yield_q_per_acre"] * bigha_factor, 2)
        profit_per_bigha_inr = round(profit_data["profit_per_acre_inr"] * bigha_factor, 0)

        # Indian units: kg (1 quintal = 100 kg)
        total_production_kg = round(profit_data["total_production_quintals"] * 100, 2)
        price_per_kg_inr    = round(profit_data["price_per_quintal"] / 100, 2)

        crop_data.append({
            "crop":                    crop,
            "suitability_conf":        conf,
            "suitability_pct":         round(conf * 100, 1),
            "is_genuine":              (idx in genuine_indices),
            "yield_q_per_bigha":       yield_q_per_bigha,
            "total_production_quintals": profit_data["total_production_quintals"],
            "price_per_quintal":       profit_data["price_per_quintal"],
            "total_production_kg":    total_production_kg,
            "price_per_kg_inr":       price_per_kg_inr,
            "estimated_sale_quantity_kg": total_production_kg,
            "gross_revenue_inr":       profit_data["gross_revenue_inr"],
            "input_cost_inr":          profit_data["input_cost_inr"],
            "net_profit_inr":          profit_data["net_profit_inr"],
            "profit_per_bigha_inr":    profit_per_bigha_inr,
            "roi_pct":                 profit_data["roi_pct"],
            "risk_score":              risk_score,
            "risk_label":              risk_label,
            "disease_risks":           diseases,
            "prevention_measures":     prevention,
            "crop_suggestions":        crop_suggestion,
            "data_confidence":         profit_data["data_confidence"],
        })

    # ---------------------------------------------------------------------------
    # Land-size filter: exclude crops that require more space than the user has
    # (e.g. sugarcane needs 2+ acres; pulses work on 0.1 acres)
    # ---------------------------------------------------------------------------
    def _min_land_for_crop(crop: str) -> float:
        return CROP_MIN_LAND_ACRES.get(crop.strip().lower(), DEFAULT_MIN_LAND_ACRES)

    crop_data_filtered = [
        c for c in crop_data
        if land_size_acres >= _min_land_for_crop(c["crop"])
    ]
    if len(crop_data_filtered) >= TOP_K_CROPS:
        crop_data = crop_data_filtered
    elif len(crop_data_filtered) > 0:
        crop_data = crop_data_filtered
    # else: keep all (filter would leave 0; show best matches with note in UI)

    # ---------------------------------------------------------------------------
    # Ranking
    # ---------------------------------------------------------------------------
    if mode == "profit":
        # Sort: genuine-gate crops first (by profit), then relaxed crops (by profit)
        genuine = [c for c in crop_data if c.get("is_genuine", True)]
        relaxed = [c for c in crop_data if not c.get("is_genuine", True)]
        genuine_ranked = rank_by_profit(genuine)
        relaxed_ranked = rank_by_profit(relaxed)
        ranked = genuine_ranked + relaxed_ranked
        for i, c in enumerate(ranked, 1):
            c["rank"] = i
    elif mode == "suitability":
        # Top 5 strongest matches for the region: rank by suitability (ML confidence) only.
        # Use crop name as tie-breaker so equal suitability always gives the same order.
        ranked = sorted(
            crop_data,
            key=lambda x: (-x["suitability_conf"], x["crop"]),
        )
        for i, c in enumerate(ranked, 1):
            c["rank"] = i
    else:
        # Balanced: normalise all three signals then compute weighted score
        confs    = [c["suitability_conf"] for c in crop_data]
        profits  = [c["net_profit_inr"]   for c in crop_data]
        risks    = [c["risk_score"]        for c in crop_data]

        norm_s = _normalise_list(confs)
        norm_p = _normalise_list(profits)
        norm_r = _normalise_list(risks)

        for i, c in enumerate(crop_data):
            c["final_score"] = _balanced_score(norm_s[i], norm_p[i], norm_r[i])

        ranked = sorted(
            crop_data,
            key=lambda x: (-x.get("final_score", 0), x["crop"]),
        )
        for i, c in enumerate(ranked, 1):
            c["rank"] = i

    # Trim to TOP_K_CROPS and ensure rank field exists
    ranked = ranked[:TOP_K_CROPS]
    for i, c in enumerate(ranked, 1):
        c.setdefault("rank", i)

    # ---------------------------------------------------------------------------
    # Global explanation (feature importance / SHAP)
    # ---------------------------------------------------------------------------
    importance_dict = metadata.get("feature_importance")
    if not importance_dict and hasattr(model, "feature_importances_"):
        importance_dict = dict(zip(feature_names, model.feature_importances_.tolist()))
    if X_test_sample is not None and y_test_sample is not None and not importance_dict:
        importance_dict = get_importance_dict(model, X_test_sample, y_test_sample, feature_names)
    importance_dict = importance_dict or {}

    explanation = explain_prediction_with_importance(model, X_scaled, feature_names, importance_dict)
    shap_text   = explain_prediction_shap_text(model, X_scaled, feature_names, classes)
    if shap_text:
        explanation = explanation + " " + shap_text

    # ---------------------------------------------------------------------------
    # Soil health messages (global, for top crop)
    # ---------------------------------------------------------------------------
    soil_health_messages = get_soil_health_messages(fd)
    best_crop            = ranked[0]["crop"]
    crop_suggestions     = get_crop_specific_suggestions(best_crop, fd)

    return {
        "top5":                  ranked,
        "explanation":           explanation,
        "soil_health_messages":  soil_health_messages,
        "crop_suggestions":      crop_suggestions,
        "land_size_acres":       land_size_acres,
        "land_size_bigha":       land_size_bigha,
        "bigha_factor":          bigha_factor,
        "scoring_mode":          mode,
        "region":                {"state": state or "Not specified", "district": district or "Not specified"},
    }
