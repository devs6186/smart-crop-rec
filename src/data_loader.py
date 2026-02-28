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
    CROP_YIELD_FNAME,
    get_state_soil_climate,
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


# Map crop_yield.csv "Crop" names to normalized training labels
# This normalizes names like "Cotton(lint)" -> "cotton", "Gram" -> "chickpea", etc.
CROP_YIELD_NAME_MAP = {
    "cotton(lint)": "cotton",
    "gram": "chickpea",
    "arhar/tur": "pigeonpeas",
    "moong(green gram)": "mungbean",
    "urad": "blackgram",
    "masoor": "lentil",
    "moth": "mothbeans",
    "cowpea(lobia)": "cowpea",
    "peas & beans (pulses)": "peas",
    "rapeseed &mustard": "mustard",
    "coconut ": "coconut",
}


def _normalize_crop_name(crop_name: str) -> str:
    """Normalize crop name from crop_yield.csv to a clean label."""
    key = (crop_name or "").strip().lower()
    return CROP_YIELD_NAME_MAP.get(key, key)


def load_crop_yield_as_training(base_df: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """
    Load ALL crops from crop_yield.csv and convert to training format.
    Uses state-based soil/climate defaults for N,P,K,temperature,humidity,ph.
    Scales Annual_Rainfall to training range (20-300).
    Returns training rows for ALL 55+ crops (not just the 22 in Crop_Recommendation.csv).
    """
    path = RAW_DATA_DIR / CROP_YIELD_FNAME
    if not path.exists():
        return None
    try:
        cy = pd.read_csv(path)
    except Exception:
        return None
    cols = [c.strip() for c in cy.columns]
    cy = cy.set_axis(cols, axis=1)
    if "Crop" not in cy.columns or "Annual_Rainfall" not in cy.columns:
        return None

    # Scale Annual_Rainfall: crop_yield range ~300-6500 -> training 20-300
    r_min = max(1, cy["Annual_Rainfall"].min()) if cy["Annual_Rainfall"].min() > 0 else 301.0
    r_max = cy["Annual_Rainfall"].max() if cy["Annual_Rainfall"].max() > 0 else 6552.0

    def scale_rainfall(x):
        if r_max <= r_min:
            return 100.0
        return round(20 + (300 - 20) * (x - r_min) / (r_max - r_min), 2)

    # If we have base_df, use per-crop medians for known crops
    known_medians = {}
    if base_df is not None and len(base_df) > 0:
        known_medians = base_df.groupby(TARGET_COLUMN)[FEATURE_COLUMNS].median().to_dict("index")

    rows = []
    for _, r in cy.iterrows():
        crop_raw = str(r.get("Crop", ""))
        label = _normalize_crop_name(crop_raw)
        if not label or label in ("oilseeds total", "other cereals", "other kharif pulses",
                                   "other rabi pulses", "other summer pulses", "other oilseeds",
                                   "other  rabi pulses"):
            continue  # skip aggregate categories

        ar = r.get("Annual_Rainfall")
        if pd.isna(ar) or ar is None:
            continue
        rain = scale_rainfall(float(ar))
        rain = max(20, min(300, rain))

        state = str(r.get("State", "")).strip()

        # Get soil/climate: use known crop medians if available, else state defaults
        if label in known_medians:
            med = known_medians[label]
            soil = {
                "N": med["N"], "P": med["P"], "K": med["K"],
                "temperature": med["temperature"], "humidity": med["humidity"], "ph": med["ph"],
            }
        else:
            soil = get_state_soil_climate(state)

        # Add small variation based on state hash to create diversity
        h = hash((state, label)) % 100
        delta = (h - 50) / 100.0  # -0.5 to +0.5

        rows.append({
            "N": round(soil["N"] + delta * 10, 1),
            "P": round(soil["P"] + delta * 8, 1),
            "K": round(soil["K"] + delta * 8, 1),
            "temperature": round(soil["temperature"] + delta * 3, 2),
            "humidity": round(soil["humidity"] + delta * 5, 2),
            "ph": round(soil["ph"] + delta * 0.5, 2),
            "rainfall": rain,
            TARGET_COLUMN: label,
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


def get_all_crop_data_paths() -> list[Path]:
    """
    Return paths to all CSVs in data/raw/ that have the crop-recommendation schema
    (N, P, K, temperature, humidity, ph, rainfall, and label/crop).
    Main file first, then others alphabetically. Use this to merge lots of data.
    """
    required = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    candidates = []
    for p in RAW_DATA_DIR.iterdir():
        if not p.suffix.lower() == ".csv" or not p.is_file():
            continue
        try:
            peek = pd.read_csv(p, nrows=1)
            peek = _normalize_columns(peek)
            if TARGET_COLUMN in peek.columns and set(FEATURE_COLUMNS).issubset(peek.columns):
                candidates.append(p)
        except Exception:
            continue
    # Prefer main file first, then sample, then rest alphabetically
    main = RAW_DATA_DIR / RAW_DATA_FNAME
    sample = RAW_DATA_DIR / SAMPLE_DATA_FNAME
    ordered = []
    if main in candidates:
        ordered.append(main)
    if sample in candidates and sample not in ordered:
        ordered.append(sample)
    for p in sorted(candidates):
        if p not in ordered:
            ordered.append(p)
    return ordered


def load_crop_data(csv_path: Path | None = None, merge_all_compatible: bool = False) -> pd.DataFrame:
    """
    Load crop recommendation dataset from CSV.
    Expects columns: N, P, K, temperature, humidity, ph, rainfall, and one of (label, crop, Crop).
    Drops rows with missing values in features or target.

    If merge_all_compatible is True, loads and concatenates all CSVs in data/raw/ that have
    this schema (so you can add more files alongside Crop_Recommendation.csv and retrain on all).

    NOTE: crop_yield.csv is NOT merged into training data — it uses a different schema
    (yield/production) and is only used for profit/region calculations.
    """
    if merge_all_compatible:
        paths = get_all_crop_data_paths()
        if not paths:
            raise FileNotFoundError(
                f"No compatible CSV found in {RAW_DATA_DIR}. "
                f"Files must have columns: {FEATURE_COLUMNS + [TARGET_COLUMN]}"
            )
        dfs = []
        for path in paths:
            df = pd.read_csv(path)
            df = _normalize_columns(df)
            if TARGET_COLUMN not in df.columns:
                continue
            missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
            if missing:
                continue
            use_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
            dfs.append(df[use_cols].dropna())
        if not dfs:
            raise ValueError(
                f"No CSV in {RAW_DATA_DIR} had required columns: {FEATURE_COLUMNS + [TARGET_COLUMN]}"
            )
        base = pd.concat(dfs, ignore_index=True)
        return base
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
