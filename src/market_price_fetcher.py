"""
data.gov.in Market Price API Fetcher
=====================================
Source: Variety-wise Daily Market Prices Data of Commodity
API   : GET https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24

Usage (from project root):
    python -m src.market_price_fetcher --api-key YOUR_KEY
    python -m src.market_price_fetcher --api-key YOUR_KEY --state Punjab
    python -m src.market_price_fetcher --api-key YOUR_KEY --max-records 5000

Or call from code:
    from src.market_price_fetcher import fetch_and_save
    fetch_and_save(api_key="YOUR_KEY")

API key:
    Get your free key at https://data.gov.in → My Account → API Keys
    The sample key (579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b)
    returns maximum 10 records per call and is NOT sufficient for bulk download.

Output:
    Saves data/raw/market_prices.csv (appends to existing cache by default).
    The file is immediately used by region_data_loader.py for price lookups.
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.config import RAW_DATA_DIR, MARKET_PRICE_FNAME

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------
API_BASE_URL   = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"
SAMPLE_API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
PAGE_SIZE      = 100      # records per API call (with a real key)
REQUEST_DELAY  = 0.5      # seconds between calls (rate limiting)
REQUEST_TIMEOUT = 30      # seconds

# Columns we keep from the API response
KEEP_COLS = ["state", "district", "market", "commodity", "variety",
             "arrival_date", "min_price", "max_price", "modal_price"]


def _is_sample_key(api_key: str) -> bool:
    return api_key.strip() == SAMPLE_API_KEY


def _make_request(api_key: str, offset: int, limit: int, state_filter: str | None) -> dict:
    """
    Make one paginated GET request to the data.gov.in API.
    Returns parsed JSON dict.
    """
    params: dict = {
        "api-key": api_key,
        "format":  "json",
        "offset":  offset,
        "limit":   limit,
    }
    if state_filter:
        # data.gov.in supports simple field filtering via filters[field]=value
        params["filters[state]"] = state_filter

    resp = requests.get(API_BASE_URL, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_all_records(
    api_key: str,
    state_filter: str | None = None,
    max_records: int = 50_000,
) -> pd.DataFrame:
    """
    Fetch market price records from data.gov.in API with pagination.

    Parameters
    ----------
    api_key : str
        Your data.gov.in API key.  Register at https://data.gov.in.
        The sample key only returns 10 records total — not useful for bulk.
    state_filter : str or None
        Optional state name to filter (e.g. "Punjab").  Reduces download size.
    max_records : int
        Safety cap on total records to fetch (default 50,000).

    Returns
    -------
    pd.DataFrame with columns: state, district, market, commodity, variety,
                                arrival_date, min_price, max_price, modal_price
    """
    if _is_sample_key(api_key):
        log.warning(
            "You are using the SAMPLE API key which returns at most 10 records. "
            "Register at https://data.gov.in → My Account → API Keys to get your own key."
        )

    # First call: get total record count
    log.info("Fetching API metadata (limit=1)...")
    meta = _make_request(api_key, offset=0, limit=1, state_filter=state_filter)
    total = int(meta.get("total", 0))

    if total == 0:
        log.warning("API returned 0 total records. Check your API key and filters.")
        return pd.DataFrame(columns=KEEP_COLS)

    log.info("Total records available: %d. Will fetch up to %d.", total, min(total, max_records))

    all_records: list[dict] = []
    offset = 0
    limit  = min(PAGE_SIZE, max_records)

    while offset < min(total, max_records):
        log.info("Fetching offset=%d / %d ...", offset, total)
        data = _make_request(api_key, offset=offset, limit=limit, state_filter=state_filter)
        records = data.get("records", [])
        if not records:
            log.info("No more records returned at offset=%d. Stopping.", offset)
            break
        all_records.extend(records)
        offset += len(records)

        # Rate limiting
        time.sleep(REQUEST_DELAY)

    log.info("Fetched %d records total.", len(all_records))
    if not all_records:
        return pd.DataFrame(columns=KEEP_COLS)

    df = pd.DataFrame(all_records)

    # Normalise column names (API may return lowercase or with spaces)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Keep only relevant columns (tolerate missing ones)
    existing = [c for c in KEEP_COLS if c in df.columns]
    df = df[existing].copy()

    # Numeric prices
    for col in ["min_price", "max_price", "modal_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["modal_price"])
    return df


def save_to_csv(df: pd.DataFrame, output_path: Path, append: bool = True) -> Path:
    """
    Save fetched records to CSV.
    If append=True and file exists, merge with existing data and deduplicate.
    """
    if append and output_path.exists():
        existing = pd.read_csv(output_path, low_memory=False)
        combined = pd.concat([existing, df], ignore_index=True)
        # Deduplicate on key fields if they exist
        dedup_cols = [c for c in ["state", "district", "commodity", "arrival_date"] if c in combined.columns]
        if dedup_cols:
            combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
        df = combined
        log.info("Merged with existing cache. Total rows: %d", len(df))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Saved %d rows to %s", len(df), output_path)
    return output_path


def fetch_and_save(
    api_key: str,
    state_filter: str | None = None,
    max_records: int = 50_000,
    append: bool = True,
    output_path: Path | None = None,
) -> Path:
    """
    Full pipeline: fetch from API → process → save to data/raw/market_prices.csv.

    Parameters
    ----------
    api_key     : Your data.gov.in API key.
    state_filter: Optionally filter by state name to reduce data volume.
    max_records : Max records to download (default 50,000).
    append      : If True, merge with existing CSV cache.
    output_path : Custom output path (defaults to data/raw/market_prices.csv).

    Returns
    -------
    Path to the saved CSV file.
    """
    out = output_path or (RAW_DATA_DIR / MARKET_PRICE_FNAME)
    log.info("Starting market price fetch from data.gov.in API...")
    df = fetch_all_records(api_key, state_filter=state_filter, max_records=max_records)
    if df.empty:
        log.warning("No data fetched. market_prices.csv not updated.")
        return out
    return save_to_csv(df, out, append=append)


def get_data_status() -> dict:
    """Return status of the local market prices cache."""
    path = RAW_DATA_DIR / MARKET_PRICE_FNAME
    if not path.exists():
        return {"exists": False, "rows": 0, "path": str(path)}
    try:
        df   = pd.read_csv(path, low_memory=False)
        rows = len(df)
        states = df["state"].nunique() if "state" in df.columns else 0
        crops  = df["commodity"].nunique() if "commodity" in df.columns else 0
        return {
            "exists":  True,
            "rows":    rows,
            "states":  states,
            "crops":   crops,
            "path":    str(path),
        }
    except Exception as exc:
        return {"exists": True, "rows": -1, "error": str(exc), "path": str(path)}


# ---------------------------------------------------------------------------
# CLI entrypoint: python -m src.market_price_fetcher --api-key KEY
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Fetch Agmarknet market prices from data.gov.in API and save to CSV."
    )
    parser.add_argument("--api-key",     required=True, help="Your data.gov.in API key")
    parser.add_argument("--state",       default=None,  help="Filter by state (optional)")
    parser.add_argument("--max-records", type=int, default=50_000, help="Max records to fetch")
    parser.add_argument("--no-append",   action="store_true", help="Overwrite cache instead of merging")
    args = parser.parse_args()

    out_path = fetch_and_save(
        api_key=args.api_key,
        state_filter=args.state,
        max_records=args.max_records,
        append=not args.no_append,
    )
    status = get_data_status()
    print(f"\nDone. market_prices.csv: {status['rows']} rows | {status.get('states', '?')} states | {status.get('crops', '?')} commodities")
    print(f"Saved to: {out_path}")
    print("\nNext: restart the Streamlit app — it will auto-load the updated prices.")
