"""
Full ML pipeline: train multiple classifiers, tune hyperparameters,
select the best model by CV stability and test F1/accuracy, and save artifacts.
"""

import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import (
    MODELS_DIR,
    FIGURES_DIR,
    MODEL_ARTIFACT_NAME,
    SCALER_ARTIFACT_NAME,
    ENCODER_ARTIFACT_NAME,
    METADATA_FNAME,
    RANDOM_STATE,
    CV_FOLDS,
    ensure_dirs,
)
from src.evaluate import evaluate_model, cross_validate_model, plot_learning_curve
from src.explainer import get_importance_dict, plot_feature_importance_bar


# -----------------------------------------------------------------------------
# Model definitions and hyperparameter grids
# -----------------------------------------------------------------------------
def get_models_and_params():
    """
    Returns a list of (name, model_instance, param_grid) for GridSearchCV.
    Models: Decision Tree, Random Forest, KNN, SVM, Logistic Regression.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return [
        (
            "Decision Tree",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
        (
            "Random Forest",
            RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
            {
                "n_estimators": [50, 100, 150],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
        ),
        (
            "KNN",
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 11, 15, 21],
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
        ),
        (
            "SVM",
            SVC(random_state=RANDOM_STATE, probability=True),
            {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto", 0.01, 0.1],
                "kernel": ["rbf"],
            },
        ),
        (
            "Logistic Regression",
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
            {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs"],
                "multi_class": ["multinomial"],
            },
        ),
    ]


def train_and_select_best(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    cv_folds=CV_FOLDS,
):
    """
    For each model: run GridSearchCV on training set, then evaluate on test set.
    Select best model by: primary = test F1-macro, tie-break = CV stability (lower std).
    Returns: best_model, best_name, scaler, label_encoder, metadata dict, comparison table.
    """
    ensure_dirs()
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, model, param_grid in get_models_and_params():
        print(f"  Grid search: {name} ...")
        gs = GridSearchCV(
            model,
            param_grid,
            cv=skf,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)
        best_estimator = gs.best_estimator_
        cv_metrics = cross_validate_model(best_estimator, X_train, y_train, cv=cv_folds)
        test_metrics = evaluate_model(best_estimator, X_test, y_test)
        results.append({
            "name": name,
            "model": best_estimator,
            "cv_accuracy_mean": cv_metrics["cv_accuracy_mean"],
            "cv_accuracy_std": cv_metrics["cv_accuracy_std"],
            "cv_f1_mean": cv_metrics["cv_f1_mean"],
            "cv_f1_std": cv_metrics["cv_f1_std"],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1_macro"],
        })

    # Select best: highest test F1; if tie, prefer lower CV F1 std (more stable)
    results.sort(
        key=lambda r: (r["test_f1"], -r["cv_f1_std"]),
        reverse=True,
    )
    best = results[0]
    best_name = best["name"]
    best_model = best["model"]

    comparison = [
        {
            "model": r["name"],
            "test_accuracy": round(r["test_accuracy"], 4),
            "test_f1_macro": round(r["test_f1"], 4),
            "cv_f1_mean": round(r["cv_f1_mean"], 4),
            "cv_f1_std": round(r["cv_f1_std"], 4),
        }
        for r in results
    ]

    # Feature importance for explainability (using test set for permutation if needed)
    importance_dict = get_importance_dict(
        best_model, X_test, y_test, feature_names
    )
    plot_feature_importance_bar(
        importance_dict,
        title=f"Feature importance ({best_name})",
        save_path=FIGURES_DIR / "feature_importance.png",
    )
    metadata = {
        "best_model_name": best_name,
        "test_accuracy": round(best["test_accuracy"], 4),
        "test_f1_macro": round(best["test_f1"], 4),
        "cv_f1_mean": round(best["cv_f1_mean"], 4),
        "cv_f1_std": round(best["cv_f1_std"], 4),
        "train_size": int(len(X_train)),   # used by UI to warn on small datasets
        "feature_names": feature_names,
        "feature_importance": {k: round(v, 6) for k, v in importance_dict.items()},
        "comparison": comparison,
    }

    return best_model, best_name, metadata, comparison


def save_artifacts(model, scaler, label_encoder, metadata, feature_names):
    """Save model, scaler, label encoder, and metadata to models/."""
    ensure_dirs()
    joblib.dump(model, MODELS_DIR / MODEL_ARTIFACT_NAME)
    joblib.dump(scaler, MODELS_DIR / SCALER_ARTIFACT_NAME)
    joblib.dump(label_encoder, MODELS_DIR / ENCODER_ARTIFACT_NAME)
    with open(MODELS_DIR / METADATA_FNAME, "w") as f:
        json.dump(metadata, f, indent=2)
    return MODELS_DIR
