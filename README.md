# Explainable AI-Based Network Intrusion Detection System (NIDS)

An end-to-end, explainable intrusion detection system that classifies network
traffic into 11 categories on the **CICIDS2017** benchmark, explains every
prediction, and serves it all through a live analyst dashboard.

Built as a B.Tech Computer Engineering capstone — from raw data to a deployed,
explainable service.

![Analyst Dashboard](reports/figures/dashboard.png)

## Highlights

- **Stacking ensemble** — Random Forest + XGBoost base learners with a Logistic
  Regression meta-learner.
- **Leakage-free class-imbalance handling** — SMOTE applied *strictly within each
  cross-validation fold*, not globally, to avoid contaminating the meta-learner.
- **Per-prediction explainability** — SHAP TreeExplainer surfaces the exact flow
  features behind every alert.
- **Deployed, not just trained** — FastAPI inference API, SQLite alert logging,
  and a real-time Streamlit dashboard.

## Results (held-out test set, 11 classes)

| Model | Accuracy | Macro-F1 | Bot recall |
|-------|:--------:|:--------:|:----------:|
| Random Forest (baseline) | 0.998 | 0.956 | 0.499 |
| XGBoost (baseline) | 0.999 | 0.969 | 0.752 |
| **Stacking + within-fold SMOTE** | **0.999** | **0.976** | **0.844** |

The within-fold SMOTE strategy lifts minority-class detection (Bot recall
0.50 → 0.84) while holding majority-class performance, and does so without the
data leakage that global SMOTE introduces. Per-alert SHAP explanation adds only
~9 ms on top of ~37 ms inference.

## Architecture

```
Raw CICIDS2017 flow
      │
      ▼
Preprocessing  (clean → drop leakage features → scale)
      │
      ▼
Stacking ensemble          SHAP TreeExplainer
  RF + XGBoost  ──────────► (per-alert feature attribution)
  → LogisticRegression
      │
      ▼
FastAPI backend  ──►  SQLite alert log
      │
      ▼
Streamlit analyst dashboard  (live feed · attack distribution · SHAP · latency)
```

## Tech stack

Python · scikit-learn · XGBoost · imbalanced-learn · SHAP · FastAPI · SQLite · Streamlit · Plotly

## Project structure

```
notebooks/   step-by-step development (inventory → preprocessing → models → SHAP)
src/         reusable modules (preprocessing, models, evaluation, explainability)
api/         FastAPI inference backend
dashboard/   Streamlit analyst dashboard
```

## Running it

The dataset and trained model are **not** committed (large / licensed). To
reproduce:

1. **Install**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows  (source .venv/bin/activate on Linux/Mac)
   pip install -r requirements.txt
   ```
2. **Get the data** — download CICIDS2017 (MachineLearningCVE, 8 CSVs) and set
   the path in `src/config.py`.
3. **Train** — run the notebooks in order (`01` → `05`). This produces the saved
   model and preprocessing artifacts.
4. **Serve**
   ```bash
   python -m uvicorn api.main:app           # terminal 1 — API at :8000
   python -m streamlit run dashboard/app.py # terminal 2 — dashboard at :8501
   ```

## Dataset

CICIDS2017 (Canadian Institute for Cybersecurity), MachineLearningCVE version —
~2.83M flows, 78 features. The two smallest classes (Heartbleed, Infiltration;
11 and 36 samples) were excluded as too sparse for reliable training and
evaluation, giving a clean 11-class problem.

## Note

Developed as an academic capstone project. This is a research/demonstration
system, not a production-hardened security appliance.
