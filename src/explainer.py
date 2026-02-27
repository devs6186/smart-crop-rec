"""
Explainability: feature importance (model-based and permutation), SHAP,
and human-readable explanation generation for "why did the model choose this crop?"
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.config import FIGURES_DIR, FEATURE_COLUMNS, ensure_dirs


def get_feature_importance(model, feature_names: list) -> dict | None:
    """
    Extract feature importance from tree-based models (DT, RF).
    Returns dict {feature_name: importance} or None if not available.
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        return dict(zip(feature_names, imp.tolist()))
    return None


def permutation_importance_sklearn(model, X, y, feature_names: list, n_repeats=10):
    """
    Compute permutation importance using sklearn (works for any model).
    X, y: numpy arrays (e.g. test set).
    Returns dict {feature_name: importance_mean}.
    """
    from sklearn.inspection import permutation_importance
    ri = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    return dict(zip(feature_names, ri.importances_mean.tolist()))


def get_importance_dict(model, X_test, y_test, feature_names: list) -> dict:
    """
    Prefer model's feature_importances_ if available; else use permutation importance.
    Returns dict {feature_name: importance} (non-negative, can be normalized).
    """
    imp = get_feature_importance(model, feature_names)
    if imp is not None:
        return imp
    return permutation_importance_sklearn(model, X_test, y_test, feature_names)


def explain_prediction_shap(model, X_row, feature_names: list, class_names: list):
    """
    For a single prediction, get SHAP values (if shap is installed).
    X_row: 2D array of shape (1, n_features).
    Returns dict {feature_name: shap_value} for the predicted class, or empty dict if SHAP unavailable.
    """
    try:
        import shap
    except ImportError:
        return {}
    # Prefer TreeExplainer for tree models (faster and exact)
    if hasattr(model, "predict_proba") and hasattr(model, "feature_importances_"):
        try:
            explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
            shap_vals = explainer.shap_values(X_row)
            if isinstance(shap_vals, list):
                pred_class = model.predict(X_row)[0]
                vals = shap_vals[pred_class][0]
            else:
                vals = shap_vals[0]
            return dict(zip(feature_names, vals.tolist()))
        except Exception:
            pass
    # Fallback: KernelExplainer (slower)
    try:
        explainer = shap.KernelExplainer(model.predict_proba, X_row)
        shap_vals = explainer.shap_values(X_row, nsamples=50)
        pred_class = model.predict(X_row)[0]
        if isinstance(shap_vals, list):
            vals = shap_vals[pred_class][0]
        else:
            vals = shap_vals[0]
        return dict(zip(feature_names, vals.tolist()))
    except Exception:
        return {}


def explain_prediction_with_importance(
    model,
    X_row: np.ndarray,
    feature_names: list,
    importance_dict: dict,
    top_n: int = 5,
) -> str:
    """
    Build a short human-readable explanation using feature importance and feature values.
    Does not require SHAP. Uses importance_dict (e.g. from get_importance_dict) to list
    top contributing factors; then describes whether the input value is high/low for that feature.
    """
    if not importance_dict:
        return "Explanation not available (no feature importance)."
    order = sorted(importance_dict.keys(), key=lambda f: importance_dict[f], reverse=True)
    top_features = order[:top_n]
    parts = []
    for f in top_features:
        idx = feature_names.index(f) if f in feature_names else None
        if idx is None:
            continue
        val = float(X_row.flat[idx])
        if f in ("temperature", "humidity", "rainfall", "ph", "N", "P", "K"):
            if val > 60 and f in ("humidity",):
                parts.append(f"high {f} ({val:.1f})")
            elif val > 30 and f == "temperature":
                parts.append(f"high {f} ({val:.1f}°C)")
            elif val < 20 and f == "temperature":
                parts.append(f"moderate-to-low {f} ({val:.1f}°C)")
            else:
                parts.append(f"{f} = {val:.1f}")
        else:
            parts.append(f"{f} = {val:.1f}")
    return "The recommendation is strongly influenced by: " + "; ".join(parts) + "."


def explain_prediction_shap_text(
    model,
    X_row: np.ndarray,
    feature_names: list,
    class_names: list,
) -> str:
    """
    Build explanation from SHAP values if available: list top positive and negative drivers.
    """
    shap_dict = explain_prediction_shap(model, X_row, feature_names, class_names)
    if not shap_dict:
        return ""
    sorted_shap = sorted(shap_dict.items(), key=lambda x: x[1], reverse=True)
    positive = [f for f, v in sorted_shap if v > 0][:3]
    negative = [f for f, v in sorted_shap if v < 0][:3]
    parts = []
    if positive:
        parts.append("Factors favouring this crop: " + ", ".join(positive) + ".")
    if negative:
        parts.append("Factors that could favour others: " + ", ".join(negative) + ".")
    return " ".join(parts) if parts else "SHAP values available but no short summary generated."


def plot_feature_importance_bar(importance_dict: dict, title: str = "Feature importance", save_path: Path | None = None):
    """Bar plot of feature importance; save to reports/figures/ if save_path not given."""
    import matplotlib.pyplot as plt
    ensure_dirs()
    path = save_path or (FIGURES_DIR / "feature_importance.png")
    names = list(importance_dict.keys())
    values = list(importance_dict.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, values, color="steelblue", alpha=0.8)
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    return path
