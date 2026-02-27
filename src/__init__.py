"""
Smart Crop Recommendation System — core package.
"""

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    ensure_dirs,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "MODELS_DIR",
    "FIGURES_DIR",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "RANDOM_STATE",
    "ensure_dirs",
]
