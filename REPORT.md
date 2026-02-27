# Smart Crop Recommendation System: A Machine Learning Approach for Agricultural Decision Support

**Report (mini research paper)** — Final Year Project

---

## Abstract

Selecting the right crop for given soil and climatic conditions is critical for yield and farmer income. We present a **Smart Crop Recommendation System** that uses supervised learning to recommend the top three most suitable crops from seven inputs: nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, soil pH, and rainfall. We compare five classifiers—Decision Tree, Random Forest, K-Nearest Neighbors, Support Vector Machine, and Logistic Regression—using stratified cross-validation and hold-out test evaluation. The best model is chosen by test F1-macro and CV stability. The system provides **explainability** (feature importance and text explanations) and **soil health interpretation** with actionable suggestions (e.g. low potassium warning for banana). A Streamlit web interface allows users to input conditions and receive recommendations with confidence scores. Results show that the chosen model achieves strong performance on the evaluation set and that **rainfall, humidity, and N/P/K** are among the most influential factors for crop suitability, supporting the system’s use as a decision-support tool in agriculture.

**Keywords:** crop recommendation, classification, explainability, feature importance, soil health, decision support.

---

## 1. Introduction

### 1.1 Background

Agriculture depends on matching crops to local soil and climate. Wrong choices lead to lower yields, wasted inputs, and economic loss. Farmers and advisors need tools that not only suggest crops but also explain *why* a crop is suitable and what constraints (e.g. low potassium, acidic soil) might limit performance.

### 1.2 Objective

We aim to build a **review-ready** system that:

1. Recommends the **top 3 crops** with **confidence scores** from seven inputs (N, P, K, temperature, humidity, pH, rainfall).  
2. Explains which factors drive the recommendation (explainability).  
3. Provides **soil health interpretation** and **crop-specific suggestions** (e.g. fertilizer or pH advice).  
4. Is **reproducible** (one-command pipeline), **modular** (separate data, train, explain, predict), and **evaluated** with proper ML practices (stratified split, CV, multiple models, overfitting checks).

### 1.3 Scope

The system is trained on a crop recommendation dataset (e.g. Kaggle: N, P, K, temperature, humidity, ph, rainfall, crop label). It is designed for academic evaluation and demonstration; deployment in production would require validation on local agronomic data and possibly expert review of thresholds and messages.

---

## 2. Related Work / Literature

Crop recommendation has been approached with rule-based systems (expert rules on soil and climate), statistical models, and machine learning. ML methods can capture non-linear relationships and interactions between soil and climate variables. Decision trees and random forests are often used for interpretability; SVMs and KNN are common for classification. Recent emphasis is on **explainable AI** so that farmers and advisors understand why a model recommends a given crop. Our work aligns with this by combining multiple classifiers, automatic model selection, feature importance, and human-readable explanations with soil health and suggestion messages.

---

## 3. Methodology

### 3.1 Data

- **Features:** N, P, K (soil nutrients), temperature (°C), humidity (%), soil pH, rainfall (mm).  
- **Target:** Crop label (multiclass).  
- **Source:** Crop Recommendation Dataset (e.g. Kaggle; 2200+ samples, 22 crops). A small sample dataset is included for runnability without download.

### 3.2 Exploratory Data Analysis (EDA)

We perform:

- **Distributions:** Histograms of each feature.  
- **Class balance:** Sample count per crop (imbalance ratio).  
- **Correlations:** Correlation matrix of the seven features.  
- **Outliers:** IQR-based counts per feature.  
- **Hypothesis:** N, P, K, rainfall, and humidity are expected to be strong drivers of crop choice; EDA and later feature importance validate this.

Plots are saved in `reports/figures/` for the report and README.

### 3.3 Preprocessing

- **Target:** Crop names encoded with `LabelEncoder` (saved for inference).  
- **Features:** Only the seven numeric inputs; no target leakage.  
- **Scaling:** `StandardScaler` fitted on the **training set only**; applied to train and test.  
- **Split:** **Stratified** train–test split (e.g. 80–20) so that class distribution is preserved. For very small datasets, the test fraction is adjusted so that each class has at least one test sample.

### 3.4 Model Training and Selection

- **Models:** Decision Tree, Random Forest, KNN, SVM (RBF), Logistic Regression.  
- **Tuning:** GridSearchCV with stratified 5-fold CV; scoring = F1-macro.  
- **Selection:** The model with the **highest test F1-macro** is chosen; if tied, the one with **lower CV F1 standard deviation** (more stable) is preferred.  
- **Overfitting:** Learning curves (train vs validation F1 vs training size) are plotted; hyperparameter search and CV reduce overfitting.

### 3.5 Explainability

- **Feature importance:** From tree-based models when available; otherwise **permutation importance** on the test set. Stored in `models/metadata.json` and displayed in the UI.  
- **Explanation text:** Generated from the top important features and their values (e.g. “The recommendation is strongly influenced by: rainfall; humidity; N; …”).  
- **Optional:** SHAP (if installed) for instance-level explanations.

### 3.6 Soil Health and Suggestions

Rule-based modules interpret input ranges (low/medium/high) for N, P, K, pH, rainfall, temperature, and humidity. They produce:

- **Soil health messages:** e.g. “Low potassium detected. Banana yield may be poor. Consider fertilizer before planting.”  
- **Crop-specific suggestions:** For the top recommended crop, messages tailored to that crop and current conditions.

### 3.7 Prediction Interface

- **API:** `predict_crop(N, P, K, temperature, humidity, ph, rainfall)`  
- **Output:** Top 3 crops with confidence %, explanation string, soil health messages, and crop suggestions.  
- **UI:** Streamlit app with sliders, Predict button, result display, and feature importance visualization.

---

## 4. Results

- **Model comparison:** All five models are evaluated on the same stratified split. Typical outcomes (with the sample or full dataset): one of KNN/SVM/Random Forest often leads on test F1; Decision Tree and Logistic Regression serve as baselines.  
- **Best model:** Selected automatically and saved with its test accuracy and F1-macro; comparison table is stored in `metadata.json`.  
- **Feature importance:** Permutation or tree-based importance shows which of N, P, K, temperature, humidity, pH, and rainfall matter most; often **rainfall, humidity, and nutrients** rank highly, consistent with agronomic intuition.  
- **Learning curve:** Saved figure shows whether the chosen model is overfitting (large train–validation gap) or underfitting (both low); tuning and CV aim for a stable validation curve.

*Exact numbers depend on the dataset (sample vs full Kaggle). The pipeline is designed to report metrics and figures in `models/metadata.json` and `reports/figures/`.*

---

## 5. Conclusion

We implemented a complete **Smart Crop Recommendation System** that:

- Uses a **full ML pipeline** (EDA, preprocessing, scaling, stratified split, CV, multiple models, hyperparameter tuning, learning curves).  
- **Compares five classifiers** and selects the best by test F1 and CV stability.  
- Provides **explainability** (feature importance and text explanations) and **soil health / crop-specific suggestions**.  
- Exposes a **prediction API** and a **Streamlit UI** for top-3 recommendations with confidence and explanations.

The system is modular, reproducible (one-command pipeline), and documented for academic review and viva. It demonstrates how ML can support agricultural decisions while remaining interpretable and actionable.

---

## 6. Future Work

- **Larger and regional data:** Train on larger and region-specific datasets for better generalization.  
- **Uncertainty:** Confidence intervals or Bayesian approaches for confidence scores.  
- **More features:** Soil type, elevation, season, irrigation.  
- **SHAP integration:** Ship SHAP in requirements and add SHAP plots to the UI.  
- **Expert validation:** Agronomist review of thresholds and suggestion messages.  
- **Deployment:** Package as an API or mobile-friendly app for field use.

---

*This report accompanies the codebase and documentation in README.md and docs/ARCHITECTURE.md.*
