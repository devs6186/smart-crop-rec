"""
Data loading and validation for the Crop Recommendation dataset.
Loads CSV from data/raw/, normalizes column names, and validates feature columns.
- Supports both 'label' and 'crop' as target column name for compatibility with different dataset versions.
- Drops rows with missing values to avoid downstream errors in scaling and model training.
"""

import pandas as pd
from pathlib import Path

from src.config import (
    RAW_DATA_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TARGET_ALIASES,
    RAW_DATA_FNAME,
    SAMPLE_DATA_FNAME,
)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure we have 'label' as target and standard feature names."""
    cols = [c.strip() for c in df.columns]
    df = df.set_axis(cols, axis=1)
    # Map common variants to our expected names
    renames = {}
    for a in TARGET_ALIASES:
        if a in df.columns and a != TARGET_COLUMN:
            renames[a] = TARGET_COLUMN
            break
    if renames:
        df = df.rename(columns=renames)
    return df


def get_data_path() -> Path:
    """Return path to main CSV; if missing, return sample path if it exists."""
    main = RAW_DATA_DIR / RAW_DATA_FNAME
    if main.exists():
        return main
    sample = RAW_DATA_DIR / SAMPLE_DATA_FNAME
    if sample.exists():
        return sample
    return main  # caller will get FileNotFoundError with clear message


def load_crop_data(csv_path: Path | None = None) -> pd.DataFrame:
    """
    Load crop recommendation dataset from CSV.
    Expects columns: N, P, K, temperature, humidity, ph, rainfall, and one of (label, crop, Crop).
    Drops rows with missing values in features or target.
    """
    path = csv_path or get_data_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Please place 'Crop_Recommendation.csv' in data/raw/ (e.g. from Kaggle: "
            "atharvaingle/crop-recommendation-dataset), or run the project's data preparation step."
        )
    df = pd.read_csv(path)
    df = _normalize_columns(df)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column not found. Expected one of {TARGET_ALIASES}. Got: {list(df.columns)}"
        )
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}. Available: {list(df.columns)}")
    # Use only required columns and drop rows with missing values
    use_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = df[use_cols].dropna()
    return df
