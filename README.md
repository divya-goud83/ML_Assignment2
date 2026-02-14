# Bank Marketing – ML Assignment 2

## Problem Statement
Predict whether a client will subscribe to a term deposit (`y` ∈ {no, yes}) using the **UCI Bank Marketing** dataset.

## Dataset
- Source: UCI Machine Learning Repository – Bank Marketing (we use `bank-additional-full.csv`, ~41k rows, **20 inputs**).
- The `duration` attribute is removed to prevent data leakage.
- UCI page: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Data files (within ZIP): https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip

## Models Implemented
1. Logistic Regression  
2. Decision Tree  
3. k-Nearest Neighbors  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

## Evaluation Metrics
For each model: **Accuracy, AUC, Precision, Recall, F1, MCC**.

A comparison table is produced at `models/metrics_summary.csv`. Per-model confusion matrices & classification reports are in `models/reports/`.

## Repository Structure
```
project/
├─ train_and_save_models.py
├─ streamlit_app.py
├─ requirements.txt
└─ models/               # created after training
   ├─ <model>.joblib
   ├─ metrics_summary.csv
   ├─ schema.json
   └─ reports/
```

## How to Run (Locally)
```bash
# 1) Create venv & install deps
pip install -r requirements.txt

# 2) Train & save models
python train_and_save_models.py

# 3) Launch Streamlit app
streamlit run streamlit_app.py
```

## Deployment (Streamlit Community Cloud)
1. Push this repo to GitHub.  
2. Go to https://streamlit.io/cloud → **New app** → select repo → branch → `streamlit_app.py` → **Deploy**.  

## Notes / Choices
- Excluded `duration` to prevent post‑hoc leakage.  
- Used `class_weight` for some models to handle class imbalance.
