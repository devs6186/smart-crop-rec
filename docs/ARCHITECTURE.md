# Smart Crop Recommendation System — Technical Architecture

## 1. Problem Statement

Farmers need to choose crops that match their local soil (N, P, K, pH) and climate (temperature, humidity, rainfall). A wrong choice leads to lower yield and economic loss. This system recommends the **top 3 most suitable crops** with confidence scores and **explainable reasons**, acting as a decision-support tool for agriculture.

---

## 2. Technical Architecture (High-Level)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE (Streamlit)                       │
│  Input: N, P, K, Temp, Humidity, pH, Rainfall → Predict → Top 3 + Explain│
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PREDICTION MODULE (src/predictor.py)                │
│  • Load trained model + scaler + metadata                                │
│  • predict_crop() → top 3 crops, confidence %, explanation               │
│  • Soil health interpretation + suggestion messages                      │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXPLAINABILITY (src/explainer.py)                     │
│  • Feature importance (model-based / permutation)                        │
│  • SHAP or equivalent for "why this crop?"                              │
│  • Human-readable explanation generation                                │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    ML PIPELINE (src/train.py, src/preprocess.py)         │
│  Data → Preprocess → Scale → Stratified Split → CV → Train → Evaluate    │
│  Models: DT, RF, KNN, SVM, Logistic Regression → Select Best             │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATA & EDA (data/, notebooks/, src/eda.py)            │
│  Raw CSV → EDA (distributions, correlations, balance) → Clean dataset   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Design choice**: Layered architecture keeps UI, prediction, explainability, and training decoupled. Training runs offline; the app only loads artifacts (model, scaler, feature names) for fast inference and explanation.

---

## 3. Folder Structure

```
SMART CROP REC/
├── data/                    # Raw and processed data
│   ├── raw/                 # Original CSV (crop_recommendation)
│   └── processed/           # Train/test splits or cleaned CSV (optional)
├── models/                  # Saved artifacts
│   ├── model.joblib         # Best trained model
│   ├── scaler.joblib        # Fitted StandardScaler
│   ├── label_encoder.joblib # Crop label encoder
│   └── metadata.json        # Feature names, metrics, model name
├── reports/                 # Generated outputs
│   └── figures/             # EDA plots, learning curves, importance plots
├── src/                     # Core Python modules
│   ├── __init__.py
│   ├── config.py            # Paths, constants, random seed
│   ├── data_loader.py       # Load CSV, validate columns
│   ├── preprocess.py        # Clean, scale, stratified split
│   ├── eda.py               # EDA scripts and plot generation
│   ├── train.py             # Full pipeline: train multiple models, select best
│   ├── evaluate.py         # Metrics, learning curves, CV stability
│   ├── explainer.py         # Feature importance, SHAP, text explanations
│   ├── predictor.py         # predict_crop(), soil health, suggestions
│   └── soil_health.py       # Rules for low/high N, P, K, pH → messages
├── app.py                   # Streamlit entry point
├── run_pipeline.py          # One-command: EDA → train → save model
├── requirements.txt
├── README.md
├── REPORT.md                # Mini research paper
└── docs/
    └── ARCHITECTURE.md      # This file
```

**Design choices**:
- **data/raw** and **data/processed**: Keeps raw data immutable; processed data can be regenerated.
- **models/**: All inference artifacts in one place; `metadata.json` stores model name and metrics for the UI/report.
- **reports/figures/**: Central place for EDA and evaluation figures (README/REPORT can reference them).
- **src/**: Modular scripts so faculty can follow “data → preprocess → train → explain → predict” in code.
- **run_pipeline.py**: Single entry point to reproduce EDA + training + saving (academic reproducibility).

---

## 4. ML Workflow Pipeline

### 4.1 Data Loading and Validation
- Load CSV from `data/raw/` (columns: N, P, K, temperature, humidity, ph, rainfall, label).
- Validate types and ranges; drop rows with missing or invalid values.
- **Why**: Ensures pipeline does not fail silently on bad data.

### 4.2 Exploratory Data Analysis (EDA)
- **Distributions**: Histograms/KDE for each feature (by crop where useful).
- **Class balance**: Bar plot of crop counts; report imbalance ratio.
- **Correlations**: Correlation matrix (numeric features); relation to target (e.g. mean N/P/K per crop).
- **Outliers**: IQR or Z-score; document and optionally cap (or flag for ablation).
- **Feature importance hypothesis**: From domain (e.g. “N/P/K and rainfall likely drive crop choice”); to be validated later by model importance and SHAP.
- **Output**: Plots saved under `reports/figures/`.

**Why**: EDA justifies preprocessing (e.g. scaling), model choice, and interpretation in the report.

### 4.3 Preprocessing
- **Target**: Encode crop names with `LabelEncoder`; save encoder to `models/`.
- **Features**: Select only N, P, K, temperature, humidity, ph, rainfall (no target leakage).
- **Scaling**: `StandardScaler` (zero mean, unit variance) fitted on training set only; applied to train and test.
- **Split**: Stratified train-test split (e.g. 80–20) so that crop distribution is preserved.
- **Why**: Scaling for SVM/KNN; stratification for stable metrics and fair comparison.

### 4.4 Model Training and Selection
- **Models**: Decision Tree, Random Forest, K-Nearest Neighbors, SVM (RBF), Logistic Regression.
- **Procedure**:
  - For each model: stratified K-fold cross-validation (e.g. 5-fold) for accuracy and F1-macro.
  - Hyperparameter tuning: GridSearchCV or RandomizedSearchCV on the training set only.
  - Refit best estimator on full training set; evaluate once on held-out test set.
- **Selection**: Compare mean CV accuracy and F1; prefer model with good CV stability (low std) and best test F1/accuracy. Document “why this model” (accuracy, F1, stability, interpretability).
- **Why**: Multiple models and CV reduce overfitting and selection bias; F1 handles possible class imbalance.

### 4.5 Overfitting Prevention
- **Cross-validation**: Reported metrics are CV means ± std.
- **Hyperparameter tuning**: Only on train set; test set used only for final evaluation.
- **Learning curves**: Plot train vs validation score vs training size; show in REPORT and optionally in README.
- **Why**: Demonstrates rigor expected in an academic project.

### 4.6 Explainability
- **Feature importance**: From tree-based model (e.g. Random Forest) or permutation importance for any model.
- **SHAP**: TreeExplainer for RF/DT or KernelExplainer for others; summary plot and/or bar plot of mean |SHAP|.
- **Explanation text**: “Model chose crop X because: high rainfall, moderate N, …” using top positive/negative SHAP features or importance.
- **Why**: Answers the viva question: “Why did the model choose rice?”

### 4.7 Prediction Module (Inference)
- **API**: `predict_crop(N, P, K, temperature, humidity, ph, rainfall)`.
- **Returns**: Top 3 crops, confidence (e.g. probability or normalized score), and a short explanation.
- **Soil health**: Rules on thresholds (e.g. low K) → suggestion message: “Low potassium detected. Banana yield may be poor. Consider fertilizer before planting.”
- **Why**: Aligns with “decision support” and “human-readable explanation” requirements.

---

## 5. Design Choices Summary

| Decision | Reason |
|----------|--------|
| StandardScaler | SVM and KNN are distance-based; LR benefits from scaled features. |
| Stratified split | Preserves crop distribution; avoids one crop missing in test. |
| F1-macro + accuracy | F1 handles imbalance; accuracy is intuitive for presentation. |
| Save scaler + encoder | Inference must apply same transform and decode labels. |
| SHAP + feature importance | SHAP is theory-grounded; importance is simple for reporting. |
| Top 3 + confidence | Gives alternatives; confidence supports “confidence-based recommendations”. |
| Soil health rules | Domain knowledge improves usefulness and explainability. |
| Single `run_pipeline.py` | One command to reproduce results (EDA → train → save). |
| Streamlit UI | Quick to build, good for demos and faculty review. |

---

## 6. Dataset

- **Source**: Crop Recommendation Dataset (e.g. Kaggle: atharvaingle/crop-recommendation-dataset) with columns: N, P, K, temperature, humidity, ph, rainfall, crop.
- **Placement**: Save as `data/raw/Crop_Recommendation.csv` (or equivalent). If the dataset is not present, `run_pipeline.py` or a data script can print instructions to download it and optionally support a fallback/sample for structure validation.
- **License**: Use in accordance with dataset license; cite in README and REPORT.

---

This architecture supports a **deployable ML product** (Streamlit app + saved model) and an **academic research project** (reproducible pipeline, EDA, multiple models, explainability, REPORT.md). All design choices are documented for viva and review.
