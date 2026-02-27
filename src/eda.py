"""
Exploratory Data Analysis for the Crop Recommendation dataset.
Generates distribution plots, class balance, correlation matrix, outlier checks,
and saves all figures to reports/figures/ for the report and README.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import FIGURES_DIR, FEATURE_COLUMNS, TARGET_COLUMN, ensure_dirs


def _setup_style():
    """Use a consistent style for all EDA figures (suitable for reports)."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 150
    plt.rcParams["font.size"] = 10


def plot_distributions(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Path:
    """
    Plot distribution of each numeric feature (histogram + KDE).
    One subplot per feature. Saves to reports/figures/feature_distributions.png
    """
    ensure_dirs()
    _setup_style()
    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    n = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)
    for i, col in enumerate(features):
        ax = axes.flat[i]
        df[col].hist(ax=ax, bins=25, edgecolor="white", alpha=0.7)
        ax.set_title(col)
        ax.set_ylabel("Count")
    for j in range(i + 1, axes.size):
        axes.flat[j].set_visible(False)
    fig.suptitle("Feature distributions (all samples)", fontsize=12, y=1.02)
    plt.tight_layout()
    out = FIGURES_DIR / "feature_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_class_balance(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> Path:
    """
    Bar plot of crop (label) counts to show class balance.
    Saves to reports/figures/class_balance.png
    """
    ensure_dirs()
    _setup_style()
    counts = df[target_col].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(counts) * 0.35)))
    counts.plot(kind="barh", ax=ax, color="steelblue", edgecolor="white")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Crop (label)")
    ax.set_title("Class balance — number of samples per crop")
    plt.tight_layout()
    out = FIGURES_DIR / "class_balance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def plot_correlation_matrix(df: pd.DataFrame) -> Path:
    """
    Correlation matrix of numeric features (and optionally relation to target via mean per class).
    Saves reports/figures/correlation_matrix.png
    """
    ensure_dirs()
    _setup_style()
    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
    ax.set_title("Correlation matrix (features)")
    plt.tight_layout()
    out = FIGURES_DIR / "correlation_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def report_outliers(df: pd.DataFrame, method: str = "iqr") -> dict:
    """
    Detect outliers using IQR or Z-score. Returns counts per feature and optional indices.
    Method 'iqr': values outside Q1 - 1.5*IQR or Q3 + 1.5*IQR.
    """
    features = [c for c in FEATURE_COLUMNS if c in df.columns]
    out = {}
    for col in features:
        x = df[col].dropna()
        if method == "iqr":
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_below = (x < low).sum()
            n_above = (x > high).sum()
        else:
            z = np.abs((x - x.mean()) / (x.std() + 1e-8))
            n_below = 0
            n_above = (z > 3).sum()
        out[col] = {"n_low": int(n_below), "n_high": int(n_above), "total": len(x)}
    return out


def plot_outlier_summary(df: pd.DataFrame) -> Path:
    """Bar plot of outlier counts per feature (IQR-based)."""
    ensure_dirs()
    _setup_style()
    outlier_counts = report_outliers(df, method="iqr")
    features = list(outlier_counts.keys())
    n_low = [outlier_counts[c]["n_low"] for c in features]
    n_high = [outlier_counts[c]["n_high"] for c in features]
    x = np.arange(len(features))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, n_low, width, label="Low (below Q1-1.5*IQR)", color="coral", alpha=0.8)
    ax.bar(x + width / 2, n_high, width, label="High (above Q3+1.5*IQR)", color="skyblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_ylabel("Number of outliers")
    ax.set_title("Outlier counts per feature (IQR method)")
    ax.legend()
    plt.tight_layout()
    out = FIGURES_DIR / "outliers_summary.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    return out


def run_full_eda(df: pd.DataFrame) -> dict:
    """
    Run all EDA steps and return paths to saved figures and a short summary.
    Use this from run_pipeline.py or a notebook.
    """
    ensure_dirs()
    paths = {}
    paths["distributions"] = plot_distributions(df)
    paths["class_balance"] = plot_class_balance(df)
    paths["correlation"] = plot_correlation_matrix(df)
    paths["outliers"] = plot_outlier_summary(df)
    outlier_report = report_outliers(df)
    n_classes = df[TARGET_COLUMN].nunique()
    n_samples = len(df)
    balance_ratio = df[TARGET_COLUMN].value_counts()
    imbalance = balance_ratio.max() / (balance_ratio.min() + 1e-8)
    summary = {
        "n_samples": n_samples,
        "n_features": len(FEATURE_COLUMNS),
        "n_classes": n_classes,
        "imbalance_ratio": round(imbalance, 2),
        "outlier_counts": outlier_report,
        "figure_paths": paths,
    }
    return summary
