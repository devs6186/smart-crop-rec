"""
Region-aware data loader for yield, market price, and cost-of-cultivation data.

Priority chain (per crop + region):
  1. District-level from CSV  →  highest confidence
  2. State-level from CSV     →  medium confidence
  3. National average (CSV)   →  low confidence
  4. Embedded fallback table  →  fallback (always available)

Embedded national averages are sourced from CACP / MIS / NHB published
statistics (approx. 2021-23) and are intentionally conservative estimates.

Expected CSV schemas (place files in data/raw/):
  state_wise_yield.csv      : State_Name, District_Name, Crop, Season,
                              Crop_Year, Area (ha), Production (tonnes)
  market_prices.csv         : State, District, Commodity, Modal_Price
  cost_of_cultivation.csv   : Crop, State, Cost_Per_Acre
  climate_vulnerability.csv : State, District, Vulnerability_Index
"""

import logging
import pandas as pd
from functools import lru_cache
from pathlib import Path

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    REGION_YIELD_FNAME,
    MARKET_PRICE_FNAME,
    COST_CULTIVATION_FNAME,
    CLIMATE_RISK_FNAME,
    UNIFIED_REGION_FNAME,
    BIGHA_TO_ACRES,
    DEFAULT_BIGHA_ACRES,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Crop-name normalisation map
# Covers common variants across government datasets and the ML label set.
# ---------------------------------------------------------------------------
CROP_NAME_MAP: dict[str, str] = {
    # cereals — generic
    "paddy": "rice", "sali": "rice", "aman": "rice", "aus": "rice",
    "boro": "rice", "kharif rice": "rice", "rabi rice": "rice",
    "corn": "maize", "makka": "maize",
    # cereals — Agmarknet specific
    "paddy(dhan)(common)": "rice",
    "paddy(dhan)(basmati)": "rice",
    "paddy(dhan)(a-grade)": "rice",
    "paddy dhan": "rice",
    # pulses — generic
    "chana": "chickpea", "gram": "chickpea", "bengal gram": "chickpea",
    "rajma": "kidneybeans", "kidney bean": "kidneybeans",
    "arhar": "pigeonpeas", "tur": "pigeonpeas", "red gram": "pigeonpeas",
    "moth": "mothbeans", "moth bean": "mothbeans",
    "moong": "mungbean", "green gram": "mungbean",
    "urad": "blackgram", "black gram": "blackgram",
    "masur": "lentil", "masoor": "lentil",
    # pulses — Agmarknet specific
    "bengal gram(gram)(whole)": "chickpea",
    "bengal gram dal (chana dal)": "chickpea",
    "arhar (tur/red gram)(whole)": "pigeonpeas",
    "arhar dal(tur dal)": "pigeonpeas",
    "pegeon pea (arhar fali)": "pigeonpeas",
    "green gram (moong)(whole)": "mungbean",
    "green gram dal (moong dal)": "mungbean",
    "black gram (urd beans)(whole)": "blackgram",
    "black gram dal (urd dal)": "blackgram",
    "lentil (masur)(whole)": "lentil",
    "masur dal": "lentil",
    "rajma (kidney beans)": "kidneybeans",
    "moth (math)": "mothbeans",
    "moth beans (matki)": "mothbeans",
    # fibre
    "kapas": "cotton", "kappas": "cotton",
    # fruits — generic
    "annar": "pomegranate",
    "kela": "banana", "plantain": "banana",
    "aam": "mango",
    "angoor": "grapes", "grape": "grapes",
    "tarbuz": "watermelon", "tarbooj": "watermelon",
    "kharbuja": "muskmelon", "melon": "muskmelon",
    "seb": "apple",
    "santra": "orange", "nagpur orange": "orange", "citrus": "orange",
    "papita": "papaya",
    "nariyal": "coconut",
    # fruits — Agmarknet specific
    "banana - green": "banana",
    "banana (raw)": "banana",
    "mango (raw-ripe)": "mango",
    "mango raw": "mango",
    "papaya (raw)": "papaya",
    "coconut seed": "coconut",
    "tender coconut": "coconut",
    "coconut oil": "coconut",
    "grapes (black)": "grapes",
    "grapes (green)": "grapes",
    "sweet orange": "orange",
    "mosambi(sweet lime)": "orange",
    # plantation
    "arabica": "coffee", "robusta": "coffee",
    "coffee (clean)": "coffee",
    "coffee beans": "coffee",
}


def _normalise_crop(name: str) -> str:
    """Lowercase and map variant spellings to the canonical ML label."""
    clean = str(name).strip().lower()
    return CROP_NAME_MAP.get(clean, clean)


# ---------------------------------------------------------------------------
# Embedded national average data
# (fallback when CSV datasets are not available)
#
# Format per crop:
#   yield_q_per_acre  : quintals per acre (1 quintal = 100 kg)
#   price_per_quintal : ₹ per quintal (approx. modal price, 2021-23)
#   cost_per_acre     : ₹ per acre (operational input cost, approx.)
# ---------------------------------------------------------------------------
CROP_NATIONAL_DEFAULTS: dict[str, dict] = {
    "rice":        {"yield_q_per_acre": 14.0, "price_per_quintal": 2183,  "cost_per_acre": 18000},
    "maize":       {"yield_q_per_acre": 11.0, "price_per_quintal": 1962,  "cost_per_acre": 14000},
    "chickpea":    {"yield_q_per_acre":  7.0, "price_per_quintal": 5230,  "cost_per_acre": 12000},
    "kidneybeans": {"yield_q_per_acre":  6.0, "price_per_quintal": 5500,  "cost_per_acre": 13000},
    "pigeonpeas":  {"yield_q_per_acre":  5.0, "price_per_quintal": 6300,  "cost_per_acre": 11000},
    "mothbeans":   {"yield_q_per_acre":  4.0, "price_per_quintal": 5000,  "cost_per_acre":  9000},
    "mungbean":    {"yield_q_per_acre":  5.0, "price_per_quintal": 7755,  "cost_per_acre": 12000},
    "blackgram":   {"yield_q_per_acre":  4.0, "price_per_quintal": 6600,  "cost_per_acre": 11000},
    "lentil":      {"yield_q_per_acre":  6.0, "price_per_quintal": 5500,  "cost_per_acre": 12000},
    "pomegranate": {"yield_q_per_acre": 60.0, "price_per_quintal": 4500,  "cost_per_acre": 35000},
    "banana":      {"yield_q_per_acre":110.0, "price_per_quintal": 1200,  "cost_per_acre": 25000},
    "mango":       {"yield_q_per_acre": 45.0, "price_per_quintal": 4000,  "cost_per_acre": 30000},
    "grapes":      {"yield_q_per_acre": 70.0, "price_per_quintal": 5000,  "cost_per_acre": 45000},
    "watermelon":  {"yield_q_per_acre":110.0, "price_per_quintal":  600,  "cost_per_acre": 20000},
    "muskmelon":   {"yield_q_per_acre": 90.0, "price_per_quintal":  800,  "cost_per_acre": 18000},
    "apple":       {"yield_q_per_acre": 45.0, "price_per_quintal": 7000,  "cost_per_acre": 40000},
    "orange":      {"yield_q_per_acre": 55.0, "price_per_quintal": 3000,  "cost_per_acre": 30000},
    "papaya":      {"yield_q_per_acre":180.0, "price_per_quintal":  500,  "cost_per_acre": 25000},
    "coconut":     {"yield_q_per_acre": 15.0, "price_per_quintal": 3000,  "cost_per_acre": 15000},
    "cotton":      {"yield_q_per_acre":  8.0, "price_per_quintal": 6620,  "cost_per_acre": 22000},
    "jute":        {"yield_q_per_acre": 18.0, "price_per_quintal": 3000,  "cost_per_acre": 18000},
    "coffee":      {"yield_q_per_acre":  5.0, "price_per_quintal": 18000, "cost_per_acre": 30000},
}


# ---------------------------------------------------------------------------
# Bigha utility
# ---------------------------------------------------------------------------

def bigha_to_acres(bigha: float, state: str | None = None) -> float:
    """
    Convert bigha to acres using state-specific conversion factor.
    Falls back to DEFAULT_BIGHA_ACRES if state is unknown.
    """
    factor = BIGHA_TO_ACRES.get(state or "", DEFAULT_BIGHA_ACRES)
    return round(bigha * factor, 4)


def acres_to_bigha(acres: float, state: str | None = None) -> float:
    """Inverse of bigha_to_acres — for display purposes."""
    factor = BIGHA_TO_ACRES.get(state or "", DEFAULT_BIGHA_ACRES)
    return round(acres / factor, 4) if factor else acres


def get_bigha_factor(state: str | None = None) -> float:
    """Return acres-per-bigha for a given state."""
    return BIGHA_TO_ACRES.get(state or "", DEFAULT_BIGHA_ACRES)


# ---------------------------------------------------------------------------
# CSV loaders — each returns None gracefully if file is absent / malformed
# ---------------------------------------------------------------------------

def _read_csv_safe(path: Path, label: str) -> pd.DataFrame | None:
    """Read a CSV; log and return None on any error."""
    if not path.exists():
        log.debug("%s dataset not found at %s — using fallback data.", label, path)
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        log.info("Loaded %s from %s (%d rows).", label, path, len(df))
        return df
    except Exception as exc:
        log.warning("Could not read %s (%s): %s", label, path, exc)
        return None


def _load_yield_df() -> pd.DataFrame | None:
    """
    Load state_wise_yield.csv.
    Expected columns: State_Name, District_Name (optional), Crop, Area, Production.
    Area in hectares, Production in tonnes.
    Returns a normalised DataFrame with columns:
        state, district, crop, yield_q_per_acre
    """
    df = _read_csv_safe(RAW_DATA_DIR / REGION_YIELD_FNAME, "yield")
    if df is None:
        return None

    # Flexible column mapping
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for key, candidates in {
        "State_Name":    ["state_name", "state", "statename"],
        "District_Name": ["district_name", "district", "districtname"],
        "Crop":          ["crop", "crop_name", "commodity"],
        "Area":          ["area", "area_ha", "area (ha)", "area(ha)"],
        "Production":    ["production", "production_tonnes", "production (tonnes)"],
    }.items():
        for c in candidates:
            if c in lower_cols:
                col_map[key] = lower_cols[c]
                break

    required = ["State_Name", "Crop", "Area", "Production"]
    if not all(k in col_map for k in required):
        log.warning("yield CSV missing required columns. Available: %s", list(df.columns))
        return None

    df = df.rename(columns={v: k for k, v in col_map.items()})
    df["Crop"] = df["Crop"].apply(_normalise_crop)
    df["State_Name"] = df["State_Name"].str.strip().str.title()
    df["District_Name"] = (
        df["District_Name"].str.strip().str.title()
        if "District_Name" in df.columns else "Unknown"
    )

    # Aggregate multiple years: take mean yield
    df["Area"]       = pd.to_numeric(df["Area"],       errors="coerce")
    df["Production"] = pd.to_numeric(df["Production"], errors="coerce")
    df = df.dropna(subset=["Area", "Production"])
    df = df[df["Area"] > 0]

    # yield in kg/ha → convert to q/acre: 1 ha = 2.471 acres, 1 q = 100 kg
    df["yield_q_per_acre"] = (df["Production"] / df["Area"]) * (1000 / 100) / 2.471

    grp_cols = ["State_Name", "District_Name", "Crop"]
    out = (
        df.groupby(grp_cols, as_index=False)["yield_q_per_acre"]
        .mean()
        .rename(columns={"State_Name": "state", "District_Name": "district", "Crop": "crop"})
    )
    return out


def _load_price_df() -> pd.DataFrame | None:
    """
    Load market_prices.csv.
    Expected columns: State, District (optional), Commodity, Modal_Price.
    Modal_Price in ₹/quintal.
    Returns normalised DataFrame with: state, district, crop, price_per_quintal
    """
    df = _read_csv_safe(RAW_DATA_DIR / MARKET_PRICE_FNAME, "prices")
    if df is None:
        return None

    lower_cols = {c.lower(): c for c in df.columns}
    col_map = {}
    for key, candidates in {
        "State":       ["state", "state_name"],
        "District":    ["district", "district_name"],
        "Commodity":   ["commodity", "crop", "crop_name"],
        "Modal_Price": ["modal_price", "modal price", "price", "avg_price", "modal_price_(rs./quintal)"],
    }.items():
        for c in candidates:
            if c in lower_cols:
                col_map[key] = lower_cols[c]
                break

    if not all(k in col_map for k in ["State", "Commodity", "Modal_Price"]):
        log.warning("price CSV missing required columns. Available: %s", list(df.columns))
        return None

    df = df.rename(columns={v: k for k, v in col_map.items()})
    df["Commodity"]   = df["Commodity"].apply(_normalise_crop)
    df["State"]       = df["State"].str.strip().str.title()
    df["District"]    = (
        df["District"].str.strip().str.title()
        if "District" in df.columns else "Unknown"
    )
    df["Modal_Price"] = pd.to_numeric(df["Modal_Price"], errors="coerce")
    df = df.dropna(subset=["Modal_Price"])

    out = (
        df.groupby(["State", "District", "Commodity"], as_index=False)["Modal_Price"]
        .mean()
        .rename(columns={
            "State": "state", "District": "district",
            "Commodity": "crop", "Modal_Price": "price_per_quintal",
        })
    )
    return out


def _load_cost_df() -> pd.DataFrame | None:
    """
    Load cost_of_cultivation.csv.
    Expected columns: Crop, State, Cost_Per_Acre.
    Returns normalised DataFrame with: state, crop, cost_per_acre
    """
    df = _read_csv_safe(RAW_DATA_DIR / COST_CULTIVATION_FNAME, "cost")
    if df is None:
        return None

    lower_cols = {c.lower(): c for c in df.columns}
    col_map = {}
    for key, candidates in {
        "Crop":         ["crop", "crop_name", "commodity"],
        "State":        ["state", "state_name"],
        "Cost_Per_Acre": [
            "cost_per_acre", "cost per acre", "cost_a2_fl", "cost_b2",
            "cost_c2", "variable_cost", "total_cost",
        ],
    }.items():
        for c in candidates:
            if c in lower_cols:
                col_map[key] = lower_cols[c]
                break

    if not all(k in col_map for k in ["Crop", "State", "Cost_Per_Acre"]):
        log.warning("cost CSV missing required columns. Available: %s", list(df.columns))
        return None

    df = df.rename(columns={v: k for k, v in col_map.items()})
    df["Crop"]         = df["Crop"].apply(_normalise_crop)
    df["State"]        = df["State"].str.strip().str.title()
    df["Cost_Per_Acre"] = pd.to_numeric(df["Cost_Per_Acre"], errors="coerce")
    df = df.dropna(subset=["Cost_Per_Acre"])

    out = (
        df.groupby(["State", "Crop"], as_index=False)["Cost_Per_Acre"]
        .mean()
        .rename(columns={"State": "state", "Crop": "crop", "Cost_Per_Acre": "cost_per_acre"})
    )
    return out


def _load_climate_df() -> pd.DataFrame | None:
    """
    Load climate_vulnerability.csv.
    Expected columns: State, District, Vulnerability_Index (0–100).
    Returns normalised DataFrame with: state, district, vulnerability_index
    """
    df = _read_csv_safe(RAW_DATA_DIR / CLIMATE_RISK_FNAME, "climate")
    if df is None:
        return None

    lower_cols = {c.lower(): c for c in df.columns}
    col_map = {}
    for key, candidates in {
        "State":    ["state", "state_name"],
        "District": ["district", "district_name"],
        "Vuln":     [
            "vulnerability_index", "composite_vulnerability", "vulnerability",
            "overall_vulnerability", "climate_vulnerability", "index",
        ],
    }.items():
        for c in candidates:
            if c in lower_cols:
                col_map[key] = lower_cols[c]
                break

    if not all(k in col_map for k in ["State", "District", "Vuln"]):
        log.warning("climate CSV missing required columns. Available: %s", list(df.columns))
        return None

    df = df.rename(columns={v: k for k, v in col_map.items()})
    df["State"]    = df["State"].str.strip().str.title()
    df["District"] = df["District"].str.strip().str.title()
    df["Vuln"]     = pd.to_numeric(df["Vuln"], errors="coerce")

    # Normalise to 0-100 if values look like 0-1
    if df["Vuln"].max() <= 1.0:
        df["Vuln"] = df["Vuln"] * 100

    out = (
        df.groupby(["State", "District"], as_index=False)["Vuln"]
        .mean()
        .rename(columns={"State": "state", "District": "district", "Vuln": "vulnerability_index"})
    )
    return out


# ---------------------------------------------------------------------------
# Cached dataset loading (loaded once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _datasets() -> dict:
    """Load all four optional CSV datasets once; cache result."""
    return {
        "yield":   _load_yield_df(),
        "price":   _load_price_df(),
        "cost":    _load_cost_df(),
        "climate": _load_climate_df(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_region_context(crop: str, state: str | None, district: str | None) -> dict:
    """
    Return region-specific agricultural context for a given crop.

    Returns a dict with keys:
        yield_q_per_acre   (float)
        price_per_quintal  (float)
        cost_per_acre      (float)
        vulnerability_index (float, 0-100; 50 = moderate if unknown)
        data_confidence    (str: "district" | "state" | "national" | "fallback")

    Implements 4-tier priority:
        district CSV > state CSV > national CSV average > embedded fallback
    """
    crop_key   = _normalise_crop(crop)
    state_std  = (state  or "").strip().title()
    dist_std   = (district or "").strip().title()
    ds         = _datasets()

    result: dict = {
        "yield_q_per_acre":   None,
        "price_per_quintal":  None,
        "cost_per_acre":      None,
        "vulnerability_index": 50.0,   # moderate default
        "data_confidence":    "fallback",
    }

    # -- Yield --
    ydf = ds["yield"]
    if ydf is not None:
        # district level
        if dist_std and dist_std not in ("Other / Not Listed", "Other"):
            row = ydf[
                (ydf["crop"] == crop_key) &
                (ydf["state"].str.title() == state_std) &
                (ydf["district"].str.title() == dist_std)
            ]
            if not row.empty:
                result["yield_q_per_acre"] = float(row["yield_q_per_acre"].mean())
                result["data_confidence"]  = "district"
        # state level
        if result["yield_q_per_acre"] is None and state_std:
            row = ydf[
                (ydf["crop"] == crop_key) &
                (ydf["state"].str.title() == state_std)
            ]
            if not row.empty:
                result["yield_q_per_acre"] = float(row["yield_q_per_acre"].mean())
                result["data_confidence"]  = "state"
        # national level from CSV
        if result["yield_q_per_acre"] is None:
            row = ydf[ydf["crop"] == crop_key]
            if not row.empty:
                result["yield_q_per_acre"] = float(row["yield_q_per_acre"].mean())
                result["data_confidence"]  = "national"

    # -- Price --
    pdf = ds["price"]
    if pdf is not None:
        for level, dist_filter in [
            ("district", dist_std),
            ("state",    None),
            ("national", None),
        ]:
            if result["price_per_quintal"] is not None:
                break
            mask = pdf["crop"] == crop_key
            if state_std and level in ("district", "state"):
                mask &= pdf["state"].str.title() == state_std
            if level == "district" and dist_filter and dist_filter not in (
                "Other / Not Listed", "Other"
            ):
                mask &= pdf["district"].str.title() == dist_filter
            row = pdf[mask]
            if not row.empty:
                result["price_per_quintal"] = float(row["price_per_quintal"].mean())
                # update confidence to the higher tier if yield conf is already higher
                if level == "district" and result["data_confidence"] == "fallback":
                    result["data_confidence"] = "district"
                elif level == "state" and result["data_confidence"] == "fallback":
                    result["data_confidence"] = "state"

    # -- Cost --
    cdf = ds["cost"]
    if cdf is not None:
        mask = cdf["crop"] == crop_key
        if state_std:
            row = cdf[mask & (cdf["state"].str.title() == state_std)]
            if row.empty:
                row = cdf[mask]
        else:
            row = cdf[mask]
        if not row.empty:
            result["cost_per_acre"] = float(row["cost_per_acre"].mean())
            if result["data_confidence"] == "fallback":
                result["data_confidence"] = "national"

    # -- Climate vulnerability --
    vdf = ds["climate"]
    if vdf is not None:
        # district level
        if dist_std and dist_std not in ("Other / Not Listed", "Other"):
            row = vdf[
                (vdf["state"].str.title() == state_std) &
                (vdf["district"].str.title() == dist_std)
            ]
            if not row.empty:
                result["vulnerability_index"] = float(row["vulnerability_index"].mean())
        # state level fallback
        if result["vulnerability_index"] == 50.0 and state_std:
            row = vdf[vdf["state"].str.title() == state_std]
            if not row.empty:
                result["vulnerability_index"] = float(row["vulnerability_index"].mean())

    # -- Fill remaining Nones from embedded fallback --
    defaults = CROP_NATIONAL_DEFAULTS.get(crop_key, {})
    if result["yield_q_per_acre"] is None:
        result["yield_q_per_acre"] = defaults.get("yield_q_per_acre", 10.0)
    if result["price_per_quintal"] is None:
        result["price_per_quintal"] = defaults.get("price_per_quintal", 3000.0)
    if result["cost_per_acre"] is None:
        result["cost_per_acre"] = defaults.get("cost_per_acre", 20000.0)

    return result


def get_climate_vulnerability(state: str | None, district: str | None) -> float:
    """
    Return climate vulnerability index (0-100) for a state/district.
    Returns 50 (moderate) if data is unavailable.
    """
    state_std = (state or "").strip().title()
    dist_std  = (district or "").strip().title()
    vdf = _datasets()["climate"]
    if vdf is None:
        return 50.0

    if dist_std and dist_std not in ("Other / Not Listed", "Other"):
        row = vdf[
            (vdf["state"].str.title() == state_std) &
            (vdf["district"].str.title() == dist_std)
        ]
        if not row.empty:
            return float(row["vulnerability_index"].mean())

    if state_std:
        row = vdf[vdf["state"].str.title() == state_std]
        if not row.empty:
            return float(row["vulnerability_index"].mean())

    return 50.0
