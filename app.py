"""
Streamlit UI — Smart Agriculture Advisory System.
Home: state, district, land size → "Proceed to Analysis". Analysis: top 5 crops by advisory score;
production and prices in kg and ₹; no profit display. Light agricultural theme.
Run with: streamlit run app.py
"""

import json
import sys
import pandas as pd
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MODELS_DIR,
    METADATA_FNAME,
    INDIAN_STATES,
    DISTRICTS_BY_STATE,
    DEFAULT_DISTRICTS,
    FEATURE_COLUMNS,
    ensure_dirs,
)
from src.predictor import load_artifacts, predict_crop
from src.market_price_fetcher import get_data_status


# ---------------------------------------------------------------------------
# Default soil/climate: vary by state so recommendations differ by region
# (Model input is only N,P,K,temp,humidity,ph,rainfall — same input → same top crops)
# Zone values aligned with training data ranges so model sees clearly different inputs.
# ---------------------------------------------------------------------------

# Agro-climatic zone defaults — deliberately distinct (training data: rice≈high rain/humidity, grapes≈low rain/humidity, apple≈cool, etc.)
# N,P,K 0–140/145/205; temp 15–45°C; humidity 15–100%; ph 3.5–9.5; rainfall 20–300 mm
_ZONE_DEFAULTS = {
    "arid_nw":       {"N": 20, "P": 28, "K": 32, "temperature": 32.0, "humidity": 44.0, "ph": 7.6, "rainfall": 28.0},   # grapes/muskmelon-like: Rajasthan, Punjab, Haryana
    "eastern_humid": {"N": 85, "P": 42, "K": 38, "temperature": 21.0, "humidity": 90.0, "ph": 5.7, "rainfall": 295.0},  # rice/blackgram-like: WB, Odisha, Assam, NE
    "southern":      {"N": 72, "P": 48, "K": 45, "temperature": 24.0, "humidity": 80.0, "ph": 6.2, "rainfall": 195.0},   # banana/papaya-like: AP, TN, Karnataka, Telangana
    "west_coast":    {"N": 84, "P": 40, "K": 36, "temperature": 20.2, "humidity": 91.0, "ph": 5.6, "rainfall": 285.0},   # blackgram/mungbean-like: Kerala, Goa
    "central":       {"N": 52, "P": 44, "K": 41, "temperature": 25.5, "humidity": 70.0, "ph": 6.1, "rainfall": 92.0},   # chickpea/mango-like: MP, UP, Bihar, Chhattisgarh
    "himalayan":     {"N": 87, "P": 42, "K": 40, "temperature": 18.5, "humidity": 87.0, "ph": 5.9, "rainfall": 255.0},   # apple/lentil-like: J&K, HP, Uttarakhand, Sikkim
    "western_dry":   {"N": 40, "P": 36, "K": 40, "temperature": 27.0, "humidity": 59.0, "ph": 6.9, "rainfall": 55.0},   # cotton/coffee-like: Gujarat, Maharashtra interior
}
_STATE_ZONE = {
    "Rajasthan": "arid_nw", "Haryana": "arid_nw", "Punjab": "arid_nw", "Delhi": "arid_nw", "Chandigarh": "arid_nw",
    "West Bengal": "eastern_humid", "Odisha": "eastern_humid", "Assam": "eastern_humid",
    "Arunachal Pradesh": "eastern_humid", "Manipur": "eastern_humid", "Meghalaya": "eastern_humid",
    "Mizoram": "eastern_humid", "Nagaland": "eastern_humid", "Tripura": "eastern_humid",
    "Andhra Pradesh": "southern", "Telangana": "southern", "Karnataka": "southern", "Tamil Nadu": "southern", "Puducherry": "southern",
    "Kerala": "west_coast", "Goa": "west_coast",
    "Maharashtra": "western_dry", "Gujarat": "western_dry", "Dadra and Nagar Haveli and Daman and Diu": "western_dry",
    "Madhya Pradesh": "central", "Chhattisgarh": "central", "Uttar Pradesh": "central", "Bihar": "central", "Jharkhand": "central",
    "Himachal Pradesh": "himalayan", "Uttarakhand": "himalayan", "Jammu and Kashmir": "himalayan", "Ladakh": "himalayan", "Sikkim": "himalayan",
    "Andaman and Nicobar Islands": "eastern_humid", "Lakshadweep": "west_coast",
}


def _state_offset(state: str, feature: str) -> float:
    """Small deterministic offset per state so same zone still gets slightly different values."""
    h = hash((state, feature)) % 100
    return (h - 50) / 50.0  # -1.0 to +1.0


@st.cache_data
def get_global_soil_climate():
    """Crop-average N, P, K, etc. from dataset (used when no state selected)."""
    try:
        from src.data_loader import load_crop_data
        df = load_crop_data()
        means = df[FEATURE_COLUMNS].mean()
        return {k: float(means[k]) for k in FEATURE_COLUMNS}
    except Exception:
        return {
            "N": 50.0, "P": 50.0, "K": 50.0,
            "temperature": 25.0, "humidity": 65.0, "ph": 6.5, "rainfall": 120.0,
        }


def get_default_soil_climate(state: str | None):
    """State-specific soil/climate so ML recommendations vary by region. Values passed correctly to predictor."""
    if not state or state not in _STATE_ZONE:
        return get_global_soil_climate()
    zone = _STATE_ZONE[state]
    base = _ZONE_DEFAULTS[zone]
    # Add small state-level offset so each state gets unique inputs (more crop variety)
    out = {}
    for k in FEATURE_COLUMNS:
        v = base[k]
        delta = _state_offset(state, k)
        if k in ("temperature", "humidity", "ph"):
            v = max(15.0, min(45.0 if k == "temperature" else 100.0 if k == "humidity" else 9.5, v + delta * 2))
        elif k == "rainfall":
            v = max(20.0, min(300.0, v + delta * 15))
        else:
            v = max(0, min(205 if k == "K" else 145 if k == "P" else 140, v + int(delta * 8)))
        out[k] = round(v, 2) if isinstance(v, float) else v
    return out


def _risk_colour(label: str) -> str:
    return {"Low": "green", "Moderate": "orange", "High": "red", "Very High": "red"}.get(label, "grey")


def _confidence_text(level: str) -> str:
    icons = {"district": "📍", "state": "🗺️", "national": "🌐", "fallback": "📊"}
    return icons.get(level, "📊") + " " + level.upper()


def build_download_df(top5: list[dict]) -> pd.DataFrame:
    """Advisory-only report: no profit columns."""
    rows = []
    for c in top5:
        rows.append({
            "Rank": c["rank"],
            "Crop": c["crop"].capitalize(),
            "Suitability (%)": c["suitability_pct"],
            "Estimated Production (kg)": c["total_production_kg"],
            "Market Price (₹/kg)": c["price_per_kg_inr"],
            "Estimated Sale Quantity (kg)": c["estimated_sale_quantity_kg"],
            "Risk Score": c["risk_score"],
            "Risk Level": c["risk_label"],
            "Data Source": c["data_confidence"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dark theme
# ---------------------------------------------------------------------------

def apply_theme():
    st.markdown("""
    <style>
    /* Dark theme */
    .stApp { background: linear-gradient(180deg, #0e1117 0%, #1a1d24 50%, #0e1117 100%); }
    [data-testid="stHeader"] { background: rgba(14, 17, 23, 0.9); }
    .main .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #b8d4b8 !important; }
    p, span, label { color: #fafafa !important; }
    .stMetric label { color: #a8c8a8; }
    div[data-testid="stExpander"] { background: #262730; border-radius: 8px; border: 1px solid #3d4a3d; }
    .stButton > button { background: #2d5a2d !important; color: white !important; border-radius: 8px; }
    .stButton > button:hover { background: #3d7a3d !important; }
    [data-testid="stSidebar"] { background: #1a1d24; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Smart Agriculture Advisory System",
        page_icon="🌾",
        layout="wide",
    )
    apply_theme()
    ensure_dirs()

    st.title("🌾 Smart Agriculture Advisory System")
    st.caption("Advisory-only recommendations • Production in kg • Prices in ₹ • No profit display")

    if not (MODELS_DIR / "model.joblib").exists():
        st.error("No trained model found. Run `python run_pipeline.py` first.")
        st.stop()

    meta_path = MODELS_DIR / METADATA_FNAME
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        train_rows = meta.get("train_size", 0)
        if train_rows and train_rows < 500:
            st.warning(
                f"**Model trained on only {train_rows} rows (sample dataset).** "
                "Crop suitability predictions will be unreliable. "
                "Download the **full Crop_Recommendation.csv** (2,200 rows) from Kaggle, "
                "place it in `data/raw/`, then run `python run_pipeline.py` to retrain."
            )

    # -----------------------------------------------------------------------
    # HOME: State, District, Land size only
    # -----------------------------------------------------------------------
    st.header("Select region and land size")

    col1, col2, col3 = st.columns(3)

    with col1:
        state_raw = st.selectbox(
            "State",
            options=["— Select State —"] + INDIAN_STATES,
            index=0,
        )
    state = None if state_raw == "— Select State —" else state_raw

    if state is None:
        district_options = []
    else:
        district_options = DISTRICTS_BY_STATE.get(state, DEFAULT_DISTRICTS)

    with col2:
        district_raw = st.selectbox(
            "District",
            options=(district_options if district_options else ["Select state first"]),
            disabled=(state is None),
        )
    district = None
    if district_raw and district_raw not in (
        "Select state first",
        "Other / Not Listed",
        "Other / Not Listed (state-level data will be used)",
    ):
        district = district_raw

    with col3:
        land_size_bigha = st.number_input(
            "Land size (bigha)",
            min_value=0.1,
            max_value=500.0,
            value=2.0,
            step=0.5,
        )

    if state:
        from src.region_data_loader import get_bigha_factor
        factor = get_bigha_factor(state)
        st.caption(f"1 bigha = {factor} acres in {state}")

    # State-specific soil/climate (after state is set) so recommendations vary by region
    default_soil = get_default_soil_climate(state)

    st.divider()

    proceed = st.button("Proceed to Analysis", type="primary", use_container_width=True)

    # -----------------------------------------------------------------------
    # After "Proceed to Analysis": run prediction and show analysis
    # -----------------------------------------------------------------------
    if proceed:
        with st.spinner("Computing recommendations..."):
            try:
                result = predict_crop(
                    default_soil["N"],
                    default_soil["P"],
                    default_soil["K"],
                    default_soil["temperature"],
                    default_soil["humidity"],
                    default_soil["ph"],
                    default_soil["rainfall"],
                    land_size_bigha=land_size_bigha,
                    state=state,
                    district=district,
                    scoring_mode="balanced",  # advisory score, not profit
                )
                st.session_state["last_result"] = result
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

    result = st.session_state.get("last_result")
    if result is None:
        st.info("Select state, district and land size, then click **Proceed to Analysis**.")
        _render_sidebar()
        return

    top5 = result["top5"]
    region = result["region"]

    # -----------------------------------------------------------------------
    # Summary (no profit)
    # -----------------------------------------------------------------------
    st.subheader("Analysis results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Top recommendation", top5[0]["crop"].capitalize())
    c2.metric("Land", f"{land_size_bigha} bigha ({result['land_size_acres']:.2f} acres)")
    loc = region["state"]
    if region["district"] != "Not specified":
        loc += " / " + region["district"]
    c3.metric("Location", loc)

    # Soil nutrient distribution (crop-average reference) — only after analysis
    N, P, K = default_soil["N"], default_soil["P"], default_soil["K"]
    total_npk = N + P + K
    if total_npk > 0:
        st.subheader("Soil nutrient distribution (crop-average reference)")
        pie_df = pd.DataFrame({
            "Nutrient": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"],
            "Share": [N / total_npk, P / total_npk, K / total_npk],
        }).set_index("Nutrient")
        st.bar_chart(pie_df)

    st.divider()

    # -----------------------------------------------------------------------
    # Top 5 crops — advisory only (no profit)
    # -----------------------------------------------------------------------
    st.subheader("Top 5 recommended crops (by advisory score)")

    medals = {1: "🥇", 2: "🥈", 3: "🥉", 4: "#4", 5: "#5"}

    for c in top5:
        rank = c["rank"]
        crop = c["crop"].capitalize()
        suit = c["suitability_pct"]
        rlbl = c["risk_label"]
        conf = c["data_confidence"]

        header = f"{medals.get(rank, '#'+str(rank))}  {crop}  |  Suitability: {suit}%  |  Risk: {rlbl}"
        with st.expander(header, expanded=(rank == 1)):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Advisory summary**")
                st.markdown(f"- **Estimated production:** {c['total_production_kg']:,.2f} kg")
                st.markdown(f"- **Market price:** ₹{c['price_per_kg_inr']:,.2f} per kg")
                st.markdown(f"- **Estimated sale quantity:** {c['estimated_sale_quantity_kg']:,.2f} kg")
                st.markdown(f"- **Data source:** {_confidence_text(conf)}")

            with col2:
                st.markdown("**Risk assessment**")
                st.markdown(f"Risk score: **{c['risk_score']}/100** — :{_risk_colour(rlbl)}[{rlbl}]")
                st.progress(int(c["risk_score"]) / 100)

                if c["disease_risks"]:
                    st.markdown("**Disease / pest risks**")
                    for d in c["disease_risks"]:
                        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(d["severity"], "⚪")
                        st.markdown(
                            f"{icon} **{d['name']}** — *{d['severity']}* | ~{int(d['probability']*100)}% | {d['season']}"
                        )

            if c["prevention_measures"]:
                st.markdown("**Prevention & management**")
                for m in c["prevention_measures"]:
                    st.markdown(f"- {m}")

            if c["crop_suggestions"]:
                with st.expander("Soil-based growing tips"):
                    for s in c["crop_suggestions"]:
                        st.info(s)

    st.divider()

    if result["soil_health_messages"]:
        st.subheader("Soil & climate health notes")
        for msg in result["soil_health_messages"]:
            st.warning(msg)

    with st.expander("Why these crops? (ML explanation)"):
        st.info(result["explanation"])
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            st.caption(
                f"Model: {meta.get('best_model_name', 'N/A')} | "
                f"Test F1: {meta.get('test_f1_macro', 'N/A')} | "
                f"Trained on: {meta.get('train_size', '?')} rows"
            )

    st.divider()
    st.subheader("Download report")
    report_df = build_download_df(top5)
    loc_tag = (region["state"] or "NoRegion").replace(" ", "_")
    st.dataframe(report_df, use_container_width=True)
    st.download_button(
        label="Download CSV",
        data=report_df.to_csv(index=False).encode("utf-8"),
        file_name=f"advisory_{loc_tag}.csv",
        mime="text/csv",
    )

    if any(c["data_confidence"] == "fallback" for c in top5):
        st.info(
            "Some values use embedded national averages. "
            "For district-level accuracy, add state_wise_yield.csv and cost_of_cultivation.csv in data/raw/."
        )

    # New analysis
    if st.button("Start new analysis"):
        st.session_state.pop("last_result", None)
        st.rerun()

    _render_sidebar()


def _render_sidebar():
    sb = st.sidebar
    sb.divider()
    sb.markdown("**Data used for analysis**")
    status = get_data_status()
    if status["exists"] and status["rows"] > 0:
        sb.caption(f"{status['rows']:,} records in total used by the engine (across {status.get('states', '?')} states)")
    with sb.expander("Refresh via data.gov.in API"):
        st.write("Paste your data.gov.in API key (register at data.gov.in → My Account)")
        api_key_input = st.text_input("API Key", type="password")
        state_filter = st.selectbox("Filter by state", ["All states"] + INDIAN_STATES)
        if st.button("Fetch prices"):
            if not api_key_input:
                st.error("Enter an API key first.")
            else:
                sf = None if state_filter == "All states" else state_filter
                with st.spinner("Fetching..."):
                    try:
                        from src.market_price_fetcher import fetch_and_save
                        from src.region_data_loader import _datasets
                        fetch_and_save(api_key=api_key_input, state_filter=sf)
                        _datasets.cache_clear()
                        st.success("Updated. Run analysis again to use new prices.")
                    except Exception as exc:
                        st.error(f"Failed: {exc}")
    sb.divider()
    sb.caption("Smart Agriculture Advisory System")


if __name__ == "__main__":
    main()
