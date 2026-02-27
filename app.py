"""
Streamlit UI for the Smart Crop Recommendation System.
Run with: streamlit run app.py
"""

import json
import streamlit as st
from pathlib import Path
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, METADATA_FNAME, ensure_dirs
from src.predictor import load_artifacts, predict_crop


def main():
    st.set_page_config(
        page_title="Smart Crop Recommendation",
        page_icon="🌾",
        layout="centered",
    )
    ensure_dirs()
    st.title("🌾 Smart Crop Recommendation System")
    st.markdown(
        "Recommend the best crops based on **soil nutrients (N, P, K)**, **pH**, and **climate** (temperature, humidity, rainfall)."
    )

    # Check if model exists
    if not (MODELS_DIR / "model.joblib").exists():
        st.warning(
            "No trained model found. Run the pipeline first: `python run_pipeline.py`"
        )
        st.stop()

    # Sidebar: input sliders (ranges aligned with typical dataset)
    st.sidebar.header("Soil & climate inputs")
    N = st.sidebar.slider("Nitrogen (N)", 0, 140, 50, help="Soil nitrogen level")
    P = st.sidebar.slider("Phosphorus (P)", 0, 145, 50)
    K = st.sidebar.slider("Potassium (K)", 0, 205, 50)
    temperature = st.sidebar.slider("Temperature (°C)", 15.0, 45.0, 25.0, 0.1)
    humidity = st.sidebar.slider("Humidity (%)", 15.0, 100.0, 65.0, 0.1)
    ph = st.sidebar.slider("Soil pH", 3.5, 9.5, 6.5, 0.1)
    rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 120.0, 0.5)

    # Run prediction on first load or when user clicks Predict
    run_predict = st.sidebar.button("Predict", type="primary")
    if "last_result" not in st.session_state:
        run_predict = True
    if run_predict:
        with st.spinner("Computing recommendations..."):
            try:
                out = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
                st.session_state["last_result"] = out
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()
    out = st.session_state.get("last_result")
    if out is None:
        st.info("Adjust the sliders and click **Predict** to get crop recommendations.")
        st.stop()

        # Top 3 crops with confidence
        st.subheader("Top 3 recommended crops")
        for r in out["top3"]:
            pct = r["confidence"] * 100
            st.metric(
                label=f"#{r['rank']} — {r['crop'].capitalize()}",
                value=f"{pct:.1f}%",
                delta="confidence" if r["rank"] == 1 else None,
            )
        st.progress(out["top3"][0]["confidence"])

        # Explanation
        st.subheader("Why this recommendation?")
        st.info(out["explanation"])

        # Soil health messages
        if out["soil_health_messages"]:
            st.subheader("Soil & climate notes")
            for msg in out["soil_health_messages"]:
                st.warning(msg)

        # Crop-specific suggestions (for top crop)
        if out["crop_suggestions"]:
            st.subheader(f"Suggestions for {out['top3'][0]['crop'].capitalize()}")
            for s in out["crop_suggestions"]:
                st.success(s)

        # Feature importance (from saved metadata)
        meta_path = MODELS_DIR / METADATA_FNAME
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            imp = meta.get("feature_importance")
            if imp:
                st.subheader("Feature importance (what drives crop choice)")
                df_imp = pd.DataFrame(
                    [{"Feature": k, "Importance": v} for k, v in imp.items()]
                ).sort_values("Importance", ascending=True)
                st.bar_chart(df_imp.set_index("Feature"))
            st.caption(f"Model: {meta.get('best_model_name', 'N/A')} | Test F1: {meta.get('test_f1_macro', 'N/A')}")

    st.sidebar.markdown("---")
    st.sidebar.caption("Final Year Project — Smart Crop Recommendation")

if __name__ == "__main__":
    main()
