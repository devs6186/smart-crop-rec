# Smart Crop Recommendation System

A **final-year-project-grade** ML system that recommends the best crops for a farmer based on soil nutrients (N, P, K), soil pH, and climate (temperature, humidity, rainfall). It outputs the **top 3 crops** with **confidence scores** and **human-readable explanations**, including soil health notes and crop-specific suggestions.

---

## Problem Statement

Choosing the wrong crop for given soil and climate leads to lower yield and economic loss. This system acts as a **decision-support tool**: given seven inputs (N, P, K, temperature, humidity, pH, rainfall), it recommends the most suitable crops and explains which factors drove the recommendation (explainability). It also highlights soil/climate issues (e.g. low potassium) and suggests actions (e.g. consider fertilizer before planting banana).

---

## Methodology

1. **Data**  
   Dataset with features: N, P, K, temperature, humidity, ph, rainfall; target: crop label. (Example: [Kaggle – Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).)

2. **Exploratory Data Analysis (EDA)**  
   Distributions of each feature, class balance, correlation matrix, and outlier analysis (IQR). Plots are saved under `reports/figures/`.

3. **Preprocessing**  
   Label encoding for crops; `StandardScaler` fitted on the training set only; **stratified** train–test split to preserve class distribution.

4. **Model training and selection**  
   Five classifiers are trained and tuned via **GridSearchCV** (stratified K-fold CV):
   - Decision Tree  
   - Random Forest  
   - K-Nearest Neighbors (KNN)  
   - Support Vector Machine (SVM)  
   - Logistic Regression  

   The **best model** is chosen by **test F1-macro**; ties are broken by **CV stability** (lower CV F1 std). Hyperparameter tuning and learning curves are used to reduce overfitting.

5. **Explainability**  
   - **Feature importance**: from tree models or **permutation importance** (saved in `models/metadata.json`).  
   - **Explanation text**: top contributing features and their values so the system can answer “Why did the model choose this crop?”  
   - Optional: **SHAP** (install `shap`) for richer explanations.

6. **Soil health and suggestions**  
   Rule-based interpretation of N, P, K, pH, rainfall, temperature, humidity (e.g. low/high) and crop-specific messages (e.g. “Low potassium detected. Banana yield may be poor. Consider fertilizer before planting.”).

7. **Prediction API**  
   `predict_crop(N, P, K, temperature, humidity, ph, rainfall)` returns top 3 crops with confidence, explanation, soil health messages, and crop suggestions.

---

## Algorithms Used

| Algorithm        | Role                          |
|-----------------|--------------------------------|
| StandardScaler  | Feature scaling (zero mean, unit variance) |
| LabelEncoder    | Encode crop names to integers  |
| Decision Tree   | Baseline; interpretable         |
| Random Forest   | Robust; provides feature importance |
| KNN             | Instance-based; often strong on this data |
| SVM (RBF)       | Good separation in scaled space |
| Logistic Regression | Linear baseline; fast |
| GridSearchCV    | Hyperparameter tuning          |
| Stratified K-Fold | Cross-validation and fair comparison |
| Permutation importance / SHAP | Explainability |

---

## Results

- **Best model** is selected automatically (e.g. KNN or SVM depending on data size and split).  
- Metrics: **accuracy** and **F1-macro** on a held-out test set; **CV mean ± std** for stability.  
- **Learning curve** (train vs validation F1) is saved to `reports/figures/learning_curve.png` to illustrate over/underfitting.  
- **Feature importance** (which soil/climate factors influence crop choice most) is saved and shown in the UI.

*With the included sample dataset (small), results are illustrative. For publication-quality results, use the full Kaggle dataset (e.g. 2200+ samples).*

---

## Screenshots

After running the app, you can add screenshots here, for example:

- **App – Input sliders and top 3 recommendations**  
  *(Add: `reports/figures/screenshot_app.png` or similar)*

- **EDA – Feature distributions**  
  `reports/figures/feature_distributions.png`

- **EDA – Class balance**  
  `reports/figures/class_balance.png`

- **Feature importance**  
  `reports/figures/feature_importance.png`

- **Learning curve**  
  `reports/figures/learning_curve.png`

---

## Project Structure

```
SMART CROP REC/
├── data/raw/          # Place Crop_Recommendation.csv here (or use sample)
├── models/            # Trained model, scaler, encoder, metadata (after run_pipeline.py)
├── reports/figures/   # EDA and evaluation plots
├── src/               # Core code (config, data, preprocess, eda, train, evaluate, explainer, predictor, soil_health)
├── app.py             # Streamlit UI
├── run_pipeline.py    # One-command pipeline
├── requirements.txt
├── README.md
├── REPORT.md          # Mini research-style report
└── docs/ARCHITECTURE.md
```

---

## How to Run

### One-time setup

```bash
# Clone or download the project, then:
cd "SMART CROP REC"
pip install -r requirements.txt
```

### Get the dataset

- **Option A:** Download [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) from Kaggle and place the CSV as `data/raw/Crop_Recommendation.csv`.  
- **Option B:** Use the included sample file `data/raw/Crop_Recommendation_sample.csv` (small, for quick testing).

### Train the model (one command)

```bash
python run_pipeline.py
```

This will: load data → run EDA (save figures) → preprocess → train and compare all five models → select the best → save model and metadata to `models/` → plot learning curve.

### Run the web app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal. Use the sliders to set N, P, K, temperature, humidity, pH, and rainfall; click **Predict** to see the top 3 crops, confidence, explanation, soil health messages, and feature importance.

---

## Documentation

- **Technical design**: `docs/ARCHITECTURE.md` (architecture, folder structure, ML workflow, design choices).  
- **Academic report**: `REPORT.md` (abstract, introduction, methodology, results, conclusion, future work).  
- **Code**: Comments in `src/` explain logic for faculty review.

---

## License and Dataset

Use the dataset in accordance with its source (e.g. Kaggle) and cite it in your report. This project is for academic and educational use.
