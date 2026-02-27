"""
Model evaluation: metrics, learning curves, and cross-validation stability.
Used by the training pipeline and for report/README figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.config import FIGURES_DIR, CV_FOLDS, ensure_dirs


def _setup_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 150


def evaluate_model(model, X_test, y_test, label_encoder=None):
    """
    Compute accuracy and F1-macro on test set.
    Optionally return per-class metrics and confusion matrix.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_test, y_pred, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


def cross_validate_model(model, X, y, cv=CV_FOLDS):
    """
    Stratified K-fold CV; returns mean and std of accuracy and F1-macro.
    Used to compare models and report stability.
    """
    scoring = ["accuracy", "f1_macro"]
    results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    return {
        "cv_accuracy_mean": results["test_accuracy"].mean(),
        "cv_accuracy_std": results["test_accuracy"].std(),
        "cv_f1_mean": results["test_f1_macro"].mean(),
        "cv_f1_std": results["test_f1_macro"].std(),
    }


def plot_learning_curve(
    estimator,
    X,
    y,
    title="Learning curve",
    cv=5,
    out_path=None,
):
    """
    Plot train and validation score vs training set size.
    Helps detect overfitting (large gap) or underfitting (both low).
    """
    ensure_dirs()
    _setup_style()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="f1_macro",
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2)
    ax.plot(train_sizes, train_mean, "o-", label="Train F1")
    ax.plot(train_sizes, test_mean, "o-", label="Validation F1")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("F1 (macro)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = out_path or (FIGURES_DIR / "learning_curve.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close()
    return path
