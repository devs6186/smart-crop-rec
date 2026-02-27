"""
Streamlit UI — Smart Crop Recommendation & Profit Analysis System.
Simple analytics page: region-first flow, top-5 crops with full economic breakdown.
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
    ensure_dirs,
)
from src.predictor import load_artifacts, predict_crop
from src.market_price_fetcher import get_data_status


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_inr(val: float) -> str:
    return "Rs {:,}".format(int(val))


def _risk_colour(label: str) -> str:
    return {"Low": "green", "Moderate": "orange", "High": "red", "Very High": "red"}.get(label, "grey")


def _confidence_text(level: str) -> str:
    icons = {"district": "📍", "state": "🗺️", "national": "🌐", "fallback": "📊"}
    return icons.get(level, "📊") + " " + level.upper()


def build_download_df(top5: list[dict]) -> pd.DataFrame:
    rows = []
    for c in top5:
        rows.append({
            "Rank":                   c["rank"],
            "Crop":                   c["crop"].capitalize(),
            "Suitability (%)":        c["suitability_pct"],
            "Yield (q/bigha)":        c["yield_q_per_bigha"],
            "Total Production (q)":   c["total_production_quintals"],
            "Market Price (Rs/q)":    int(c["price_per_quintal"]),
            "Gross Revenue (Rs)":     int(c["gross_revenue_inr"]),
            "Input Cost (Rs)":        int(c["input_cost_inr"]),
            "Net Profit (Rs)":        int(c["net_profit_inr"]),
            "Profit per Bigha (Rs)":  int(c["profit_per_bigha_inr"]),
            "ROI (%)":                c["roi_pct"],
            "Risk Score":             c["risk_score"],
            "Risk Level":             c["risk_label"],
            "Data Source":            c["data_confidence"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Smart Crop Recommendation",
        page_icon="🌾",
        layout="wide",
    )
    ensure_dirs()
    st.title("🌾 Smart Crop Recommendation — Profit Analysis")
    st.caption("Top-5 crops ranked by expected profit | Region-aware | Disease risk included | All values in Indian Rupees")

    if not (MODELS_DIR / "model.joblib").exists():
        st.error("No trained model found. Run `python run_pipeline.py` first.")
        st.stop()

    # Dataset quality warning
    import joblib
    meta_path = MODELS_DIR / METADATA_FNAME
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        train_rows = meta.get("train_size", 0)
        if train_rows and train_rows < 500:
            st.warning(
                f"**Model trained on only {train_rows} rows (sample dataset).** "
                "Crop suitability predictions will be unreliable. "
                "Download the **full Crop_Recommendation.csv** (2,200 rows, 22 crops) "
                "from Kaggle (`atharvaingle/crop-recommendation-dataset`), place it in "
                "`data/raw/`, then run `python run_pipeline.py` to retrain."
            )

    sb = st.sidebar

    # -----------------------------------------------------------------------
    # STEP 1 — Region Selection (top of sidebar)
    # -----------------------------------------------------------------------
    sb.header("Step 1 — Select Your Region")

    state = sb.selectbox(
        "State",
        options=["— Select State —"] + INDIAN_STATES,
        index=0,
    )
    state = None if state == "— Select State —" else state

    if state is None:
        sb.info("Select a state to see district options.")
        district_options = []
    else:
        district_options = DISTRICTS_BY_STATE.get(state, DEFAULT_DISTRICTS)

    district_raw = sb.selectbox(
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

    # Show bigha conversion info
    if state:
        from src.region_data_loader import get_bigha_factor
        factor = get_bigha_factor(state)
        sb.caption(f"Bigha size in {state}: 1 bigha = {factor} acres")

    sb.divider()

    # -----------------------------------------------------------------------
    # STEP 2 — Soil & Climate Parameters
    # -----------------------------------------------------------------------
    sb.header("Step 2 — Soil & Climate Parameters")
    N           = sb.slider("Nitrogen (N)",      0,   140,  50)
    P           = sb.slider("Phosphorus (P)",    0,   145,  50)
    K           = sb.slider("Potassium (K)",     0,   205,  50)
    temperature = sb.slider("Temperature (C)",  15.0, 45.0, 25.0, 0.1)
    humidity    = sb.slider("Humidity (%)",     15.0, 100.0, 65.0, 0.1)
    ph          = sb.slider("Soil pH",           3.5,  9.5,  6.5, 0.1)
    rainfall    = sb.slider("Rainfall (mm)",    20.0, 300.0, 120.0, 0.5)

    sb.divider()

    # -----------------------------------------------------------------------
    # STEP 3 — Farm Details
    # -----------------------------------------------------------------------
    sb.header("Step 3 — Farm Details")
    land_size_bigha = sb.number_input(
        "Land Size (Bigha)", min_value=0.1, max_value=500.0, value=2.0, step=0.5,
    )

    scoring_mode = sb.radio(
        "Ranking mode",
        options=["profit", "balanced"],
        index=0,
        help="Profit-first: rank by expected net profit.\nBalanced: mix of suitability + profit - risk.",
    )

    sb.divider()
    run_btn = sb.button("Analyse Crops", type="primary", use_container_width=True)

    # -----------------------------------------------------------------------
    # Run prediction
    # -----------------------------------------------------------------------
    trigger = run_btn or ("last_result" not in st.session_state)
    if trigger:
        with st.spinner("Computing recommendations..."):
            try:
                result = predict_crop(
                    N, P, K, temperature, humidity, ph, rainfall,
                    land_size_bigha=land_size_bigha,
                    state=state,
                    district=district,
                    scoring_mode=scoring_mode,
                )
                st.session_state["last_result"] = result
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.stop()

    result = st.session_state.get("last_result")
    if result is None:
        st.info("Fill in the sidebar and click **Analyse Crops**.")
        st.stop()

    top5   = result["top5"]
    region = result["region"]

    # -----------------------------------------------------------------------
    # Summary banner
    # -----------------------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Crop",       top5[0]["crop"].capitalize())
    c2.metric("Max Net Profit",  _fmt_inr(top5[0]["net_profit_inr"]))
    c3.metric("Land",            f"{land_size_bigha} bigha ({result['land_size_acres']:.2f} ac)")
    loc = region["state"]
    if region["district"] != "Not specified":
        loc += " / " + region["district"]
    c4.metric("Location", loc)

    st.divider()

    # -----------------------------------------------------------------------
    # Profit comparison chart
    # -----------------------------------------------------------------------
    st.subheader("Net Profit Comparison")
    chart_df = pd.DataFrame({
        "Crop":           [c["crop"].capitalize() for c in top5],
        "Net Profit (Rs)":[c["net_profit_inr"] for c in top5],
    }).set_index("Crop")
    st.bar_chart(chart_df)

    st.divider()

    # -----------------------------------------------------------------------
    # Per-crop detailed results
    # -----------------------------------------------------------------------
    st.subheader("Top 5 Recommended Crops")

    medals = {1: "🥇", 2: "🥈", 3: "🥉", 4: "#4", 5: "#5"}

    for c in top5:
        rank   = c["rank"]
        crop   = c["crop"].capitalize()
        profit = c["net_profit_inr"]
        suit   = c["suitability_pct"]
        rlbl   = c["risk_label"]
        conf   = c["data_confidence"]

        is_genuine = c.get("is_genuine", True)
        tag = "" if is_genuine else "  [Alt]"
        header = f"{medals.get(rank, '#'+str(rank))}  {crop}{tag}  |  Profit: {_fmt_inr(profit)}  |  Suitability: {suit}%  |  Risk: {rlbl}"

        with st.expander(header, expanded=(rank == 1)):
            if not is_genuine:
                st.caption("Alt: ML confidence is low for your soil/climate — shown as a high-profit alternative for comparison.")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Economic Breakdown**")
                eco = {
                    "Expected Yield":      f"{c['yield_q_per_bigha']} q / bigha",
                    "Total Production":    f"{c['total_production_quintals']} quintals",
                    "Market Price":        f"Rs {int(c['price_per_quintal']):,} / quintal",
                    "Gross Revenue":       _fmt_inr(c["gross_revenue_inr"]),
                    "Input Cost":          _fmt_inr(c["input_cost_inr"]),
                    "Net Profit":          _fmt_inr(c["net_profit_inr"]),
                    "Profit per Bigha":    _fmt_inr(c["profit_per_bigha_inr"]),
                    "ROI":                 f"{c['roi_pct']}%",
                    "Data Source":         _confidence_text(conf),
                }
                st.table(pd.DataFrame({"Value": eco}))

            with col2:
                st.markdown("**Risk Assessment**")
                st.markdown(f"Risk Score: **{c['risk_score']}/100** — :{_risk_colour(rlbl)}[{rlbl}]")
                st.progress(int(c["risk_score"]) / 100)

                if c["disease_risks"]:
                    st.markdown("**Disease / Pest Risks:**")
                    for d in c["disease_risks"]:
                        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(d["severity"], "⚪")
                        st.markdown(
                            f"{icon} **{d['name']}**  \n"
                            f"Severity: *{d['severity']}* | ~{int(d['probability']*100)}% chance | {d['season']}"
                        )

            # Prevention measures
            if c["prevention_measures"]:
                with st.expander("Prevention & Management"):
                    for m in c["prevention_measures"]:
                        st.markdown(f"- {m}")

            # Soil tips
            if c["crop_suggestions"]:
                with st.expander("Soil-Based Growing Tips"):
                    for s in c["crop_suggestions"]:
                        st.info(s)

    st.divider()

    # -----------------------------------------------------------------------
    # Soil health
    # -----------------------------------------------------------------------
    if result["soil_health_messages"]:
        st.subheader("Soil & Climate Health Notes")
        for msg in result["soil_health_messages"]:
            st.warning(msg)

    # -----------------------------------------------------------------------
    # ML explanation
    # -----------------------------------------------------------------------
    with st.expander("Why these crops? (ML Explanation)"):
        st.info(result["explanation"])
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            imp = meta.get("feature_importance")
            if imp:
                imp_df = pd.DataFrame(
                    [{"Feature": k, "Importance": v} for k, v in imp.items()]
                ).sort_values("Importance", ascending=True)
                st.bar_chart(imp_df.set_index("Feature"))
            st.caption(
                f"Model: {meta.get('best_model_name', 'N/A')} | "
                f"Test F1: {meta.get('test_f1_macro', 'N/A')} | "
                f"Trained on: {meta.get('train_size', '?')} rows"
            )

    st.divider()

    # -----------------------------------------------------------------------
    # Download report
    # -----------------------------------------------------------------------
    st.subheader("Download Report")
    report_df  = build_download_df(top5)
    loc_tag    = (region["state"] or "NoRegion").replace(" ", "_")
    st.dataframe(report_df, use_container_width=True)
    st.download_button(
        label     = "Download CSV",
        data      = report_df.to_csv(index=False).encode("utf-8"),
        file_name = f"crop_recommendation_{loc_tag}.csv",
        mime      = "text/csv",
    )

    # -----------------------------------------------------------------------
    # Data confidence note
    # -----------------------------------------------------------------------
    if any(c["data_confidence"] == "fallback" for c in top5):
        st.info(
            "Some values use embedded national averages (FALLBACK). "
            "For district-level accuracy, place these in `data/raw/`:\n"
            "- `state_wise_yield.csv` — state/district yield data\n"
            "- `cost_of_cultivation.csv` — input cost per acre"
        )

    # -----------------------------------------------------------------------
    # Market data refresh (sidebar bottom)
    # -----------------------------------------------------------------------
    sb.divider()
    sb.markdown("**Market Price Data**")
    status = get_data_status()
    if status["exists"] and status["rows"] > 0:
        sb.caption(f"{status['rows']:,} price records | {status.get('states','?')} states")
    with sb.expander("Refresh via data.gov.in API"):
        st.write("Paste your data.gov.in API key (register free at data.gov.in → My Account)")
        api_key_input = st.text_input("API Key", type="password")
        state_filter  = st.selectbox("Filter by state", ["All states"] + INDIAN_STATES)
        if st.button("Fetch Prices"):
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
                        st.success("Updated! Click Analyse to use new prices.")
                    except Exception as exc:
                        st.error(f"Failed: {exc}")

    sb.divider()
    sb.caption("Smart Crop Recommendation System — Final Year Project")


if __name__ == "__main__":
    main()
