"""
Configuration and constants for the Smart Crop Recommendation System.
Centralizes paths, column names, and random seed for reproducibility.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Base paths (project root = parent of 'src')
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# -----------------------------------------------------------------------------
# Data file names
# -----------------------------------------------------------------------------
RAW_DATA_FNAME = "Crop_Recommendation.csv"
# Fallback: minimal sample data embedded or generated if raw file missing
SAMPLE_DATA_FNAME = "Crop_Recommendation_sample.csv"

# -----------------------------------------------------------------------------
# Feature and target column names (must match dataset)
# -----------------------------------------------------------------------------
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"  # some datasets use "crop" or "label"
# Allow alternate target name for flexibility
TARGET_ALIASES = ("label", "crop", "Crop")

# -----------------------------------------------------------------------------
# ML constants
# -----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
# For top-k recommendations
TOP_K_CROPS = 3

# -----------------------------------------------------------------------------
# Model save names
# -----------------------------------------------------------------------------
MODEL_ARTIFACT_NAME = "model.joblib"
SCALER_ARTIFACT_NAME = "scaler.joblib"
ENCODER_ARTIFACT_NAME = "label_encoder.joblib"
METADATA_FNAME = "metadata.json"

# -----------------------------------------------------------------------------
# Ensure directories exist (called when pipeline runs)
# -----------------------------------------------------------------------------
def ensure_dirs():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
