# Smart Agriculture Advisory System

An **ML-based advisory system** that recommends suitable crops for Indian farmers by **state and district**, land size, and region-specific soil/climate. It shows the **top 5 crops** by **advisory score** (suitability, risk, and regional potential), with **estimated production in kg**, **market price in ₹/kg**, **risk and disease information**, and **prevention measures**. No profit figures are shown—advisory only. Built for final-year / academic use.

---

## Features

- **Region-first flow**: Select **state**, **district**, and **land size (bigha)**. No manual soil/climate input—the system uses **state-specific agro-climatic defaults** so recommendations **vary by region** (e.g. Rajasthan vs Kerala vs Himachal Pradesh).
- **Advisory-only output**: Top 5 crops ranked by a **balanced advisory score** (suitability + regional potential − risk). No net profit, ROI, or profit charts.
- **Indian units**: Production and sale quantity in **kg**; prices in **₹/kg**; land in **bigha** (with acres shown). No quintals or tons.
- **Per-crop details**: Suitability %, estimated production (kg), market price (₹/kg), estimated sale quantity (kg), risk score, disease/pest risks, prevention measures, and soil-based growing tips.
- **Soil nutrient view**: After analysis, a soil nutrient distribution (N, P, K) chart is shown as a crop-average reference for the selected region.
- **Data used for analysis**: Sidebar shows how many **records in total** are used by the engine (e.g. market/yield data across states). Optional refresh via **data.gov.in** API.
- **Dark theme** UI; optional lighter theme in code.

---

## Problem Statement

Choosing the wrong crop for a given region and land leads to lower yield and wasted effort. This system acts as a **decision-support tool**: given state, district, and land size, it uses **region-specific soil/climate profiles** and an ML model to recommend the most suitable crops, with explainability, risk, and preventive advice—without showing direct profit to avoid misleading estimates.

---

## Methodology

1. **Data**  
   - **Crop recommendation dataset**: N, P, K, temperature, humidity, ph, rainfall → crop label (e.g. [Kaggle – Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)).  
   - **Regional data** (optional): `state_wise_yield.csv`, `market_prices.csv`, `cost_of_cultivation.csv`, `climate_vulnerability.csv` in `data/raw/` for state/district-aware yield, price, and risk. If absent, embedded national averages are used.

2. **Region-specific inputs**  
   - States are mapped to **agro-climatic zones** (arid NW, eastern humid, southern, west coast, central, Himalayan, western dry). Each zone has distinct default N, P, K, temperature, humidity, ph, rainfall (aligned with training data). A small **state-level offset** ensures different states get slightly different inputs so the ML model sees varied conditions and recommends different crops.

3. **ML pipeline**  
   - **Preprocessing**: Label encoding, `StandardScaler` on training set, stratified train–test split.  
   - **Models**: Decision Tree, Random Forest, KNN, SVM, Logistic Regression—tuned via GridSearchCV; best model by **test F1-macro** (e.g. SVM).  
   - **Prediction**: `predict_crop(N, P, K, temperature, humidity, ph, rainfall, land_size_bigha, state, district, scoring_mode="balanced")` returns top 5 crops with suitability, production (kg), price (₹/kg), risk, disease risks, prevention, and explanations.

4. **Explainability & soil health**  
   - Feature importance (in `models/metadata.json`), explanation text, and rule-based soil health messages and crop-specific suggestions.

---

## Algorithms Used

| Component        | Role |
|------------------|------|
| StandardScaler   | Feature scaling |
| LabelEncoder     | Crop labels |
| SVM / KNN / RF / etc. | Classification (best model saved) |
| GridSearchCV    | Hyperparameter tuning |
| Stratified K-Fold | Cross-validation |
| Region data loader | State/district yield, price, cost, vulnerability |
| Balanced scoring | Suitability + regional potential − risk (no profit in UI) |

---

## Project Structure

```
SMART CROP REC/
├── data/raw/              # Crop_Recommendation.csv (or sample); optional: state_wise_yield, market_prices, cost_of_cultivation, climate_vulnerability
├── models/                # model.joblib, scaler.joblib, label_encoder.joblib, metadata.json (after run_pipeline.py)
├── reports/figures/       # EDA and evaluation plots
├── src/                   # config, data_loader, preprocess, train, evaluate, predictor, region_data_loader, profit_engine, risk_engine, soil_health, explainer, market_price_fetcher
├── app.py                 # Streamlit UI — Smart Agriculture Advisory System
├── run_pipeline.py        # One-command ML pipeline
├── requirements.txt
├── README.md
├── REPORT.md
└── docs/ARCHITECTURE.md
```

---

## How to Run

### One-time setup

```bash
cd "SMART CROP REC"
pip install -r requirements.txt
```

### Dataset

- Place **Crop_Recommendation.csv** in `data/raw/` (e.g. from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)), or use **Crop_Recommendation_sample.csv** for quick testing.  
- Optional: Add `state_wise_yield.csv`, `market_prices.csv`, `cost_of_cultivation.csv`, `climate_vulnerability.csv` for better region-aware results.

### Train the model

```bash
python run_pipeline.py
```

This loads data, runs EDA, preprocesses, trains and compares models, selects the best, and saves artifacts to `models/`.

### Run the web app

```bash
streamlit run app.py
```

Or:

```bash
python -m streamlit run app.py
```

Then open the URL (e.g. http://localhost:8501). Select **state**, **district**, and **land size (bigha)** → click **Proceed to Analysis** → view top 5 crops, production (kg), price (₹/kg), risk, diseases, and prevention. Use **Start new analysis** to run again.

---

## Documentation

- **Technical design**: `docs/ARCHITECTURE.md`  
- **Academic report**: `REPORT.md`  
- **Code**: Comments in `src/` and `app.py`

---

## License and Dataset

Use the dataset in accordance with its source (e.g. Kaggle) and cite it in your report. This project is for academic and educational use.
