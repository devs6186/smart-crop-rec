# Smart Crop Advisory System

> **What does it do?** You tell it your **state**, **district**, and **land size** — it tells you the **best 5 crops** to grow, how much you can produce, current market prices, and what diseases to watch out for.

---

## The Problem

Indian farmers often pick crops based on habit or neighbors' advice, not data. A wrong crop choice for your soil, climate, and land size means lower yield, wasted money, and wasted effort.

## The Solution

This system acts as a **free, AI-powered crop advisor**. It uses machine learning trained on real agricultural data to match your region's soil and climate to the most suitable crops — and backs every recommendation with production estimates, market prices, and risk warnings.

---

## What You Get

| For each recommended crop | Example |
|---------------------------|---------|
| **Suitability score** | "This crop is 92% suitable for your region" |
| **Estimated production** | "You can produce ~4,500 kg on your land" |
| **Market price** | "Current price: ₹42/kg" |
| **Risk score** | "Risk: 35/100 (Moderate)" |
| **Disease warnings** | "Watch out for Late Blight — spray Mancozeb at 7-day intervals" |
| **Growing tips** | "Your soil is low on Potassium — consider K fertilizer" |

---

## How It Works (Simple Version)

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  YOU PROVIDE     │     │  THE SYSTEM       │     │  YOU GET            │
│                  │     │                   │     │                     │
│  • State         │────>│  • Looks up your  │────>│  • Top 5 crops      │
│  • District      │     │    region's soil   │     │  • Production (kg)  │
│  • Land size     │     │  • Runs ML model   │     │  • Price (₹/kg)     │
│    (in bigha)    │     │  • Checks risks    │     │  • Risk warnings    │
│                  │     │  • Finds prices    │     │  • Disease alerts    │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
```

**No manual soil input needed.** The system knows that Rajasthan has different soil than Kerala, and recommends accordingly.

---

## How It Works (Technical Version)

1. **Region mapping** — Each Indian state is mapped to an agro-climatic zone (arid, humid, coastal, etc.) with realistic N, P, K, temperature, humidity, pH, and rainfall defaults. District-level offsets add further variation.

2. **ML classification** — Six models (Random Forest, SVM, KNN, Decision Tree, Extra Trees, Logistic Regression) are trained on crop recommendation data and compared via GridSearchCV. The best model (by F1-macro score) is auto-selected. Current accuracy: **96%** across **51 crops**.

3. **Risk engine** — A knowledge base of **120+ crop-disease entries** (sourced from ICAR, NIPHM) computes a composite risk score combining climate vulnerability and disease severity.

4. **Market data** — Embedded national price averages, upgradeable with live data from the **data.gov.in API**.

5. **Land filter** — Crops that need more land than you have (e.g. sugarcane needs 2+ acres) are automatically excluded.

---

## Quick Start

### 1. Install

```bash
cd smart-crop-rec
pip install -r requirements.txt
```

### 2. Add data

Place `Crop_Recommendation.csv` in `data/raw/` — download from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).

### 3. Train the model

```bash
python run_pipeline.py
```

This trains 6 models, picks the best one, and saves it. Takes about 2-3 minutes.

### 4. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Select your state, district, land size → click **Proceed to Analysis**.

### 5. (Optional) Run the landing page

```bash
cd my-product-site
npm install
npm run dev
```

Open `http://localhost:3000` for the landing page with the seed-to-plant video.

---

## Project Structure

```
smart-crop-rec/
│
├── app.py                  ← The main web app (Streamlit)
├── run_pipeline.py         ← Train the ML model (run this first)
│
├── src/
│   ├── config.py           ← All settings, state lists, crop parameters
│   ├── data_loader.py      ← Loads and validates crop datasets
│   ├── preprocess.py       ← Scales features, splits train/test
│   ├── train.py            ← Trains 6 models, picks the best
│   ├── evaluate.py         ← Accuracy, F1, confusion matrix
│   ├── predictor.py        ← The prediction API (core logic)
│   ├── zone_soil.py        ← State → soil/climate defaults
│   ├── region_data_loader.py ← Yield, price, cost data by region
│   ├── profit_engine.py    ← Production and revenue calculations
│   ├── risk_engine.py      ← Disease risks + composite risk score
│   ├── soil_health.py      ← Soil health warnings and tips
│   ├── explainer.py        ← "Why this crop?" explanations
│   ├── crop_params.py      ← Agronomic parameters for 51 crops
│   └── market_price_fetcher.py ← Live prices from data.gov.in
│
├── data/raw/               ← Put your CSV datasets here
├── models/                 ← Trained model files (auto-generated)
├── reports/figures/        ← Charts and plots (auto-generated)
├── tests/                  ← Automated tests
├── my-product-site/        ← Landing page (Next.js)
└── .streamlit/config.toml  ← App theme settings
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **ML models** | scikit-learn (SVM, Random Forest, KNN, Decision Tree, Extra Trees, Logistic Regression) |
| **Data processing** | pandas, NumPy |
| **Web app** | Streamlit |
| **Landing page** | Next.js, Tailwind CSS, Framer Motion |
| **Explainability** | Feature importance, SHAP (optional) |
| **Market data** | data.gov.in REST API |

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Crops supported | 51 |
| ML accuracy | 96% (F1-macro) |
| Indian states covered | 28+ (all states and UTs) |
| Disease/pest entries | 120+ |
| Models compared | 6 |
| Market data | Embedded + live API |

---

## Screenshots

**Home** — Select state, district, and land size:

> State → District → Land size (bigha) → Proceed to Analysis

**Results** — Top 5 crops with full breakdown:

> Suitability % · Production (kg) · Price (₹/kg) · Risk score · Disease alerts · Prevention tips

---

## Running Tests

```bash
python -m pytest tests/test_crop_variety.py -v
```

Tests verify that:
- Different states get different soil/climate profiles
- Different states get different crop recommendations
- Land size affects which crops are shown
- All output fields are present and valid

---

## Data Sources

| Source | Used for |
|--------|----------|
| [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) | ML training data |
| ICAR Crop Production Technology guidelines | Agronomic parameters, disease data |
| FAO Ecocrop database | Crop growth ranges |
| NIPHM publications | Disease and pest knowledge base |
| data.gov.in API | Live market prices |
| CACP / MIS / NHB statistics | National average yields and costs |

---

## License

For academic and educational use. Cite the dataset source (Kaggle) in your report.
