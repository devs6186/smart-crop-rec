"""
Agronomic parameter database for crop recommendation training data generation.

Each crop has realistic ranges for N, P, K, temperature, humidity, ph, rainfall
based on ICAR/NBSS&LUP guidelines, FAO Ecocrop data, and Indian agricultural
extension literature.

Format per crop: {feature: (min, max, mean, std)}
The generator samples from truncated normal distributions within [min, max].
"""

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Crop parameter ranges: {crop_name: {feature: (min, max, mean, std)}}
#
# Sources:
#   - ICAR Crop Production Technology guidelines
#   - FAO Ecocrop database (ecocrop.fao.org)
#   - NBSS&LUP soil fertility maps
#   - Handbook of Agriculture (ICAR, 6th Ed.)
#   - Package of Practices for major crops (state agriculture universities)
# ──────────────────────────────────────────────────────────────────────────────

CROP_PARAMS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    # ── NEW CROPS (not in original 22-crop Kaggle dataset) ──────────────────

    # Cereals
    "wheat": {
        "N": (80, 140, 110, 12),
        "P": (40, 80, 60, 8),
        "K": (30, 60, 45, 6),
        "temperature": (12, 22, 17, 2),
        "humidity": (40, 65, 52, 5),
        "ph": (6.0, 7.8, 6.8, 0.4),
        "rainfall": (35, 85, 60, 10),
    },
    "bajra": {  # pearl millet
        "N": (40, 80, 60, 8),
        "P": (20, 50, 35, 6),
        "K": (15, 40, 28, 5),
        "temperature": (25, 38, 32, 2.5),
        "humidity": (30, 55, 42, 5),
        "ph": (6.5, 8.2, 7.3, 0.4),
        "rainfall": (20, 55, 35, 7),
    },
    "jowar": {  # sorghum
        "N": (50, 90, 70, 8),
        "P": (25, 55, 40, 6),
        "K": (20, 50, 35, 6),
        "temperature": (25, 35, 30, 2),
        "humidity": (40, 65, 52, 5),
        "ph": (6.0, 8.0, 7.0, 0.4),
        "rainfall": (30, 80, 55, 10),
    },
    "ragi": {  # finger millet
        "N": (30, 70, 50, 8),
        "P": (20, 50, 35, 6),
        "K": (20, 50, 35, 6),
        "temperature": (20, 30, 25, 2),
        "humidity": (60, 85, 72, 5),
        "ph": (5.0, 7.5, 6.2, 0.5),
        "rainfall": (60, 120, 90, 12),
    },
    "barley": {
        "N": (50, 85, 68, 7),          # lower N than mustard
        "P": (30, 55, 42, 5),
        "K": (30, 55, 42, 5),          # higher K than mustard
        "temperature": (8, 18, 13, 2), # cooler than mustard
        "humidity": (30, 50, 40, 4),   # drier than mustard
        "ph": (7.0, 8.8, 7.8, 0.4),   # more alkaline
        "rainfall": (20, 50, 35, 6),
    },

    # Pulses
    "peas": {
        "N": (10, 30, 20, 4),
        "P": (40, 80, 60, 8),
        "K": (30, 60, 45, 6),
        "temperature": (10, 22, 16, 2.5),
        "humidity": (50, 75, 62, 5),
        "ph": (6.0, 7.5, 6.8, 0.3),
        "rainfall": (40, 85, 60, 8),
    },
    "cowpea": {
        "N": (8, 25, 16, 3),        # lower N than groundnut (cowpea is more N-fixing)
        "P": (35, 70, 52, 7),
        "K": (10, 30, 20, 4),       # lower K than groundnut
        "temperature": (28, 38, 33, 2),  # more heat-tolerant than groundnut
        "humidity": (55, 80, 68, 5),
        "ph": (5.5, 7.0, 6.2, 0.3),    # slightly more acidic preference
        "rainfall": (40, 75, 55, 7),     # less rainfall than groundnut
    },

    # Oilseeds
    "groundnut": {
        "N": (20, 45, 32, 5),       # higher N need than cowpea
        "P": (30, 70, 50, 8),
        "K": (30, 60, 45, 6),       # higher K than cowpea
        "temperature": (24, 32, 28, 1.5),  # cooler than cowpea
        "humidity": (50, 72, 60, 4),
        "ph": (6.0, 7.8, 6.9, 0.4),       # slightly alkaline preference
        "rainfall": (55, 110, 80, 10),      # more rainfall than cowpea
    },
    "soyabean": {
        "N": (5, 25, 15, 4),
        "P": (40, 80, 60, 8),
        "K": (30, 60, 45, 6),
        "temperature": (22, 32, 27, 2),
        "humidity": (55, 80, 68, 5),
        "ph": (5.5, 7.2, 6.3, 0.4),
        "rainfall": (60, 120, 85, 12),
    },
    "mustard": {
        "N": (70, 110, 90, 8),         # higher N than barley
        "P": (25, 55, 40, 6),
        "K": (15, 35, 25, 4),          # lower K than barley
        "temperature": (14, 24, 19, 2), # warmer than barley
        "humidity": (50, 70, 60, 4),    # more humid than barley
        "ph": (6.0, 7.5, 6.8, 0.3),    # more neutral
        "rainfall": (30, 65, 45, 7),
    },
    "sunflower": {
        "N": (50, 90, 70, 8),
        "P": (35, 70, 52, 7),
        "K": (25, 55, 40, 6),
        "temperature": (20, 30, 25, 2),
        "humidity": (40, 65, 52, 5),
        "ph": (6.0, 8.0, 7.0, 0.4),
        "rainfall": (40, 80, 60, 8),
    },
    "sesamum": {  # sesame/til
        "N": (30, 60, 45, 6),          # higher N than castor
        "P": (25, 55, 40, 6),          # higher P
        "K": (15, 35, 25, 4),          # lower K than castor
        "temperature": (28, 40, 34, 2), # hotter than castor
        "humidity": (35, 55, 45, 4),    # drier
        "ph": (5.5, 7.5, 6.5, 0.4),
        "rainfall": (25, 55, 38, 6),    # less rainfall
    },
    "castor seed": {
        "N": (10, 30, 20, 4),          # much lower N than sesamum
        "P": (15, 40, 28, 5),          # lower P
        "K": (20, 50, 35, 6),          # higher K than sesamum
        "temperature": (22, 33, 28, 2), # cooler than sesamum
        "humidity": (40, 65, 52, 5),    # more humid
        "ph": (6.0, 8.5, 7.2, 0.5),    # more alkaline
        "rainfall": (40, 80, 60, 8),    # more rainfall
    },

    # Cash crops
    "sugarcane": {
        "N": (100, 160, 130, 12),
        "P": (30, 65, 48, 7),
        "K": (50, 100, 75, 10),
        "temperature": (25, 38, 32, 2.5),
        "humidity": (60, 85, 72, 5),
        "ph": (5.5, 8.0, 6.8, 0.5),
        "rainfall": (100, 200, 150, 18),
    },
    "tobacco": {
        "N": (50, 85, 68, 7),          # lower N than dry chillies
        "P": (20, 48, 34, 5),          # lower P
        "K": (50, 90, 70, 8),          # higher K
        "temperature": (18, 28, 23, 2), # cooler than dry chillies
        "humidity": (50, 72, 60, 4),    # less humid
        "ph": (5.0, 6.5, 5.7, 0.3),    # more acidic
        "rainfall": (45, 90, 65, 9),
    },

    # Vegetables / tubers
    "potato": {
        "N": (80, 140, 110, 12),
        "P": (50, 90, 70, 8),
        "K": (80, 140, 110, 12),
        "temperature": (12, 22, 17, 2),
        "humidity": (60, 85, 72, 5),
        "ph": (5.0, 7.0, 6.0, 0.4),
        "rainfall": (40, 80, 60, 8),
    },
    "onion": {
        "N": (80, 130, 105, 10),       # higher N than garlic
        "P": (50, 90, 70, 8),          # higher P
        "K": (50, 90, 70, 8),
        "temperature": (18, 30, 24, 2.5), # warmer than garlic
        "humidity": (55, 80, 68, 5),      # more humid
        "ph": (5.8, 7.2, 6.4, 0.3),
        "rainfall": (40, 80, 60, 8),      # more rainfall
    },
    "garlic": {
        "N": (45, 90, 68, 9),           # lower N than onion
        "P": (35, 70, 52, 7),           # lower P
        "K": (55, 100, 78, 9),          # higher K than onion
        "temperature": (10, 20, 15, 2),  # much cooler than onion
        "humidity": (40, 62, 50, 4),     # drier
        "ph": (6.2, 8.0, 7.1, 0.4),     # more alkaline
        "rainfall": (20, 50, 35, 6),     # less rainfall
    },
    "sweet potato": {
        "N": (40, 80, 60, 8),
        "P": (30, 65, 48, 7),
        "K": (60, 120, 90, 12),
        "temperature": (22, 32, 27, 2),
        "humidity": (60, 85, 72, 5),
        "ph": (5.0, 6.8, 5.8, 0.4),
        "rainfall": (60, 120, 90, 12),
    },
    "tapioca": {  # cassava
        "N": (30, 70, 50, 8),
        "P": (20, 55, 38, 7),
        "K": (60, 120, 90, 12),
        "temperature": (25, 35, 30, 2),
        "humidity": (60, 90, 75, 6),
        "ph": (5.0, 7.0, 6.0, 0.4),
        "rainfall": (80, 160, 120, 15),
    },
    "ginger": {
        "N": (70, 120, 95, 10),        # higher N than turmeric
        "P": (40, 75, 58, 7),          # higher P than turmeric
        "K": (55, 100, 78, 9),
        "temperature": (18, 27, 22, 2), # cooler than turmeric
        "humidity": (80, 97, 88, 3),    # more humid than turmeric
        "ph": (5.2, 6.5, 5.8, 0.3),    # more acidic
        "rainfall": (150, 230, 185, 15), # more rainfall
    },
    "turmeric": {
        "N": (40, 85, 62, 9),          # lower N than ginger
        "P": (20, 50, 35, 6),          # lower P than ginger
        "K": (60, 115, 88, 10),        # higher K than ginger
        "temperature": (25, 35, 30, 2), # warmer than ginger
        "humidity": (65, 85, 75, 4),    # less humid than ginger
        "ph": (5.5, 7.8, 6.6, 0.5),    # more alkaline
        "rainfall": (90, 160, 125, 13), # less rainfall
    },

    # Spices
    "dry chillies": {
        "N": (90, 140, 115, 10),       # higher N than tobacco
        "P": (40, 75, 58, 7),          # higher P
        "K": (40, 75, 58, 7),
        "temperature": (25, 35, 30, 2), # warmer than tobacco
        "humidity": (60, 85, 72, 5),    # more humid
        "ph": (5.5, 7.0, 6.2, 0.3),    # more acidic
        "rainfall": (55, 105, 80, 10),
    },
    "coriander": {
        "N": (30, 60, 45, 6),
        "P": (25, 55, 40, 6),
        "K": (20, 45, 32, 5),
        "temperature": (15, 25, 20, 2),
        "humidity": (40, 65, 52, 5),
        "ph": (6.0, 8.0, 7.0, 0.4),
        "rainfall": (30, 70, 50, 8),
    },
    "black pepper": {
        "N": (60, 110, 85, 10),
        "P": (20, 50, 35, 6),
        "K": (80, 140, 110, 12),
        "temperature": (22, 32, 27, 2),
        "humidity": (70, 95, 82, 5),
        "ph": (4.5, 6.5, 5.5, 0.4),
        "rainfall": (150, 250, 200, 18),
    },
    "cardamom": {
        "N": (50, 100, 75, 10),
        "P": (30, 65, 48, 7),
        "K": (60, 120, 90, 12),
        "temperature": (15, 25, 20, 2),
        "humidity": (75, 95, 85, 4),
        "ph": (4.5, 6.5, 5.5, 0.4),
        "rainfall": (150, 280, 210, 22),
    },

    # Fibre / misc
    "mesta": {  # kenaf / Hibiscus sabdariffa
        "N": (30, 60, 45, 6),
        "P": (15, 40, 28, 5),
        "K": (20, 45, 32, 5),
        "temperature": (24, 34, 29, 2),
        "humidity": (70, 90, 80, 4),
        "ph": (5.5, 7.0, 6.2, 0.3),
        "rainfall": (100, 180, 140, 15),
    },

    # Plantation crops
    "arecanut": {
        "N": (90, 140, 115, 10),       # higher N than black pepper
        "P": (15, 40, 28, 5),          # lower P than black pepper
        "K": (90, 150, 120, 12),       # higher K
        "temperature": (26, 36, 31, 2), # warmer than black pepper
        "humidity": (70, 90, 80, 4),    # less humid
        "ph": (5.5, 7.0, 6.2, 0.3),    # more alkaline than black pepper
        "rainfall": (120, 220, 170, 18), # less rainfall
    },
    "cashewnut": {
        "N": (15, 45, 30, 6),
        "P": (10, 35, 22, 5),
        "K": (25, 60, 42, 7),
        "temperature": (24, 36, 30, 2.5),
        "humidity": (55, 80, 68, 5),
        "ph": (5.0, 7.0, 6.0, 0.4),
        "rainfall": (80, 160, 120, 15),
    },
}

RANDOM_STATE = 42


def generate_crop_samples(
    crop: str,
    n_samples: int = 100,
    params: dict | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Generate realistic synthetic training samples for a single crop.

    Uses truncated normal sampling within the agronomically valid
    [min, max] range for each feature, centered on mean with given std.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    p = params or CROP_PARAMS.get(crop)
    if p is None:
        raise ValueError(f"No parameter definition for crop '{crop}'")

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    rows = {}
    for feat in features:
        lo, hi, mu, sigma = p[feat]
        # Sample from normal, then clip to valid range
        samples = rng.normal(mu, sigma, size=n_samples)
        samples = np.clip(samples, lo, hi)
        # Round appropriately
        if feat in ("N", "P", "K"):
            samples = np.round(samples, 0)
        elif feat == "ph":
            samples = np.round(samples, 2)
        else:
            samples = np.round(samples, 1)
        rows[feat] = samples

    rows["label"] = crop
    return pd.DataFrame(rows)


def generate_all_new_crops(
    n_samples_per_crop: int = 100,
) -> pd.DataFrame:
    """
    Generate balanced synthetic training data for all crops in CROP_PARAMS.
    Returns a DataFrame ready to merge with the original Crop_Recommendation.csv.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    frames = []
    for crop in sorted(CROP_PARAMS):
        df = generate_crop_samples(crop, n_samples_per_crop, rng=rng)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
