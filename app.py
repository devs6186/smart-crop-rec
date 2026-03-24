"""
Streamlit UI — Smart Agriculture Advisory System.
Home: state, district, land size → "Proceed to Analysis". Analysis: top 5 crops by advisory score;
production and prices in kg and ₹; no profit display. Dark theme with emerald accents.
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
# Soil/climate by region — see src.zone_soil for logic (shared with tests)
# ---------------------------------------------------------------------------

from src.zone_soil import get_default_soil_climate


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
# Dark + emerald theme (matches landing page)
# ---------------------------------------------------------------------------

def apply_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

    /* ── Base ── */
    .stApp {
        background: #0a0a0a !important;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stHeader"] {
        background: rgba(10, 10, 10, 0.85) !important;
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(52, 211, 153, 0.08);
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    /* ── Typography ── */
    h1 {
        color: #34d399 !important;
        font-weight: 900 !important;
        letter-spacing: 0.02em;
    }
    h2, h3 {
        color: #d1d5db !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em;
    }
    p, span, label, li {
        color: #d1d5db !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: rgba(52, 211, 153, 0.04);
        border: 1px solid rgba(52, 211, 153, 0.12);
        border-radius: 12px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetric"] label {
        color: rgba(52, 211, 153, 0.7) !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #34d399 !important;
        color: #0a0a0a !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #6ee7b7 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(52, 211, 153, 0.25);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: transparent !important;
        color: #34d399 !important;
        border: 1px solid rgba(52, 211, 153, 0.3) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(52, 211, 153, 0.1) !important;
        border-color: #34d399 !important;
    }

    /* ── Expanders ── */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        transition: border-color 0.3s ease;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(52, 211, 153, 0.2) !important;
    }
    div[data-testid="stExpander"] summary span {
        color: #e5e7eb !important;
        font-weight: 600 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #111318 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #34d399 !important;
    }

    /* ── Inputs ── */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: #111318 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: #e5e7eb !important;
    }
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div:focus-within {
        border-color: rgba(52, 211, 153, 0.5) !important;
        box-shadow: 0 0 0 1px rgba(52, 211, 153, 0.2);
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div {
        background: #34d399 !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255, 255, 255, 0.06) !important;
    }

    /* ── Info / Warning / Error boxes ── */
    .stAlert [data-testid="stAlertContentInfo"] {
        background: rgba(52, 211, 153, 0.06) !important;
        border-left-color: #34d399 !important;
        color: #d1d5db !important;
    }
    div[data-testid="stNotification"] {
        border-radius: 8px !important;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Bar chart ── */
    .stBarChart {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Caption ── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: rgba(255, 255, 255, 0.35) !important;
    }
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] span {
        color: rgba(255, 255, 255, 0.35) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb { background: #1a3a2a; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #2d5a3d; }
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

    st.title("🌾 Smart Crop Advisory System")
    st.caption("Advisory recommendations • Production in kg • Prices in ₹ • Region-aware intelligence")

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
            index=0,
            key=f"district_{state}",
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

    # State+district-specific soil/climate so recommendations vary by region
    default_soil = get_default_soil_climate(state, district)

    st.divider()

    st.markdown(
        """<style>
        div[data-testid="stButton"] > button[kind="primary"] {
            background: #34d399 !important;
            color: #000000 !important;
            font-size: 1.1rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.06em !important;
            padding: 0.8rem 2rem !important;
            border: none !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: #6ee7b7 !important;
            color: #000000 !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )
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
                    scoring_mode="suitability",  # top 5 strongest matches for the region
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
    st.caption("Recommendations are tailored to your land size, state, and district — crops requiring more space are excluded for small holdings.")

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
    st.subheader("Top 5 recommended crops (by suitability — strongest matches first)")

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


def _get_engine_stats():
    """ML model stats + market data stats."""
    try:
        meta_path = MODELS_DIR / METADATA_FNAME
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            train_size = meta.get("train_size", 0)
        else:
            train_size = 0
        model, _, label_encoder, _ = load_artifacts()
        num_crops = len(label_encoder.classes_) if label_encoder else 0
    except Exception:
        train_size = 0
        num_crops = "?"

    status = get_data_status()
    num_states = status.get("states", "—") if status.get("exists") else "—"

    return {"records": train_size, "states": num_states, "crops": num_crops}


def _render_sidebar():
    sb = st.sidebar
    sb.markdown(
        '<div style="text-align:center; padding: 0.5rem 0 1rem;">'
        '<span style="color:#34d399; font-weight:900; font-size:1.1rem; letter-spacing:0.1em;">SMART CROP</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    sb.divider()
    sb.markdown("**Data used for analysis**")
    stats = _get_engine_stats()
    sb.caption(f"**Total records:** {stats['records']:,}" if isinstance(stats["records"], int) else f"**Total records:** {stats['records']}")
    sb.caption(f"**States:** {stats['states']}")
    sb.caption(f"**Total crops:** {stats['crops']}")
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
    sb.caption("Smart Crop Advisory System")


if __name__ == "__main__":
    main()
