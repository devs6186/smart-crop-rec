"""
Preprocessing for the Crop Recommendation pipeline.
- StandardScaler: fitted on train only to prevent data leakage; required for SVM/KNN.
- Stratified split: keeps class distribution in train and test so metrics are representative.
- LabelEncoder: maps crop names to integers; saved and reused at inference for consistent decoding.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
)


def prepare_X_y(df: pd.DataFrame):
    """
    Extract feature matrix X and target vector y.
    X: numeric features only (N, P, K, temperature, humidity, ph, rainfall).
    y: crop labels (string).
    """
    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(str).str.strip()
    return X, y


def encode_labels(y: pd.Series, fitted_encoder: LabelEncoder | None = None):
    """
    Encode crop names to integers. If fitted_encoder is provided, use it (for test/inference).
    Returns encoded array and the encoder (fitted on the provided y if new).
    """
    if fitted_encoder is not None:
        # Handle unseen labels at inference: map to -1 or most frequent (we use 0 as fallback for unknown)
        y_enc = np.array([fitted_encoder.transform([v])[0] if v in fitted_encoder.classes_ else 0 for v in y])
        return y_enc, fitted_encoder
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Stratified train-test split so that crop distribution is preserved in both sets.
    If dataset is small (test set would have fewer samples than classes), use a smaller
    test fraction so that each class has at least one test sample.
    """
    n_classes = len(np.unique(y))
    n_samples = len(y)
    min_test = n_classes  # need at least one per class in test
    if n_samples * test_size < min_test:
        test_size = max(0.1, min_test / n_samples)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on training data only (no test leakage)."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def scale_features(X: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """Transform features using a fitted scaler."""
    return scaler.transform(X)


def preprocess_pipeline(df: pd.DataFrame):
    """
    Full preprocessing: X, y, encode, split, scale.
    Returns:
        X_train, X_test, y_train, y_test (numpy arrays, scaled),
        scaler, label_encoder
    """
    X, y_series = prepare_X_y(df)
    y, label_encoder = encode_labels(y_series, None)

    X_train, X_test, y_train, y_test = split_data(X, y)

    scaler = fit_scaler(X_train)
    X_train_scaled = scale_features(X_train, scaler)
    X_test_scaled = scale_features(X_test, scaler)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        label_encoder,
        FEATURE_COLUMNS,
    )
