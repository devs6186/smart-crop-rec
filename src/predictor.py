"""
Prediction module: load trained artifacts and expose predict_crop().
Returns top 3 crops with confidence and human-readable explanation,
plus soil health interpretation and suggestion messages.
"""

import json
import joblib
import numpy as np
from pathlib import Path

from src.config import (
    MODELS_DIR,
    FEATURE_COLUMNS,
    MODEL_ARTIFACT_NAME,
    SCALER_ARTIFACT_NAME,
    ENCODER_ARTIFACT_NAME,
    METADATA_FNAME,
    TOP_K_CROPS,
)
from src.soil_health import get_soil_health_messages, get_crop_specific_suggestions
from src.explainer import (
    get_importance_dict,
    explain_prediction_with_importance,
    explain_prediction_shap_text,
)


def load_artifacts(models_dir: Path | None = None):
    """Load model, scaler, label encoder, and metadata from models/."""
    d = models_dir or MODELS_DIR
    model = joblib.load(d / MODEL_ARTIFACT_NAME)
    scaler = joblib.load(d / SCALER_ARTIFACT_NAME)
    label_encoder = joblib.load(d / ENCODER_ARTIFACT_NAME)
    with open(d / METADATA_FNAME) as f:
        metadata = json.load(f)
    return model, scaler, label_encoder, metadata


def _feature_dict(N, P, K, temperature, humidity, ph, rainfall):
    """Build ordered dict of inputs for scaling and explanation."""
    return {
        "N": float(N),
        "P": float(P),
        "K": float(K),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "ph": float(ph),
        "rainfall": float(rainfall),
    }


def predict_crop(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    models_dir: Path | None = None,
    X_test_sample=None,
    y_test_sample=None,
):
    """
    Main prediction API.
    Returns:
        top3: list of dicts with keys: crop, confidence, rank
        explanation: str (why this crop)
        soil_health_messages: list of str
        crop_suggestions: list of str (for top recommended crop)
    """
    model, scaler, label_encoder, metadata = load_artifacts(models_dir)
    feature_names = metadata.get("feature_names", FEATURE_COLUMNS)
    fd = _feature_dict(N, P, K, temperature, humidity, ph, rainfall)
    X = np.array([[fd[c] for c in feature_names]], dtype=float)
    X_scaled = scaler.transform(X)

    # Probabilities (or decision function for SVM without probability)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
    else:
        # Fallback: one-hot for predicted class
        pred = model.predict(X_scaled)[0]
        probs = np.zeros(len(label_encoder.classes_))
        probs[pred] = 1.0
    classes = label_encoder.classes_.tolist()
    idx_sorted = np.argsort(probs)[::-1]
    top_indices = idx_sorted[:TOP_K_CROPS]
    top3 = []
    for rank, idx in enumerate(top_indices, 1):
        crop = classes[idx]
        conf = float(probs[idx])
        top3.append({"crop": crop, "confidence": conf, "rank": rank})
    best_crop = top3[0]["crop"]

    # Explanation: use saved importance from metadata, or compute from model
    importance_dict = metadata.get("feature_importance")
    if not importance_dict and hasattr(model, "feature_importances_"):
        importance_dict = dict(
            zip(feature_names, model.feature_importances_.tolist())
        )
    if X_test_sample is not None and y_test_sample is not None and not importance_dict:
        importance_dict = get_importance_dict(
            model, X_test_sample, y_test_sample, feature_names
        )
    importance_dict = importance_dict or {}
    explanation = explain_prediction_with_importance(
        model, X_scaled, feature_names, importance_dict
    )
    shap_text = explain_prediction_shap_text(
        model, X_scaled, feature_names, classes
    )
    if shap_text:
        explanation = explanation + " " + shap_text

    soil_health_messages = get_soil_health_messages(fd)
    crop_suggestions = get_crop_specific_suggestions(best_crop, fd)

    return {
        "top3": top3,
        "explanation": explanation,
        "soil_health_messages": soil_health_messages,
        "crop_suggestions": crop_suggestions,
    }
