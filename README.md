# Explainable AI-Based Network Intrusion Detection System (NIDS)

B.Tech Computer Engineering Capstone Project.

An explainable, ensemble-based, deployable NIDS evaluated end-to-end on the
**CICIDS2017** benchmark. The system combines a stacking ensemble with
leakage-free class-imbalance handling, per-prediction explanations, and a
real-time analyst dashboard.

## Locked architecture

| Component            | Choice                                                        |
| -------------------- | ------------------------------------------------------------- |
| Base learners        | Random Forest + XGBoost                                       |
| Meta-learner         | Logistic Regression                                           |
| Ensemble strategy    | Stacking (out-of-fold predictions -> meta-learner)            |
| Class imbalance      | SMOTE (k=5) applied **within** each CV fold (not globally)    |
| Explainability       | SHAP TreeExplainer (global summary + per-alert waterfall)     |
| Classification       | Multi-class (all CICIDS2017 attack categories)                |
| Backend              | FastAPI (REST inference + SHAP endpoints)                     |
| Storage              | SQLite (alert log + inference history)                        |
| Dashboard            | Streamlit (live feed, attack distribution, SHAP, latency)     |
| Dataset              | CICIDS2017 — MachineLearningCVE 8-CSV version                 |

## Research gaps this project addresses

1. No stacking + SHAP integration on CICIDS2017 in the reviewed literature.
2. No end-to-end deployable stacking NIDS (stacking + SHAP + FastAPI + dashboard).
3. SMOTE applied within stacking CV folds (leakage-free) — the core novelty.
4. Multi-class evaluation (most high-accuracy CICIDS2017 papers are binary-only).
5. SHAP overhead measured separately from inference latency.
6. Reproducibility — CICIDS2017 used exclusively.

## Folder layout

```
nids-capstone/
├── data/            CICIDS2017 CSVs live here (NOT committed — see data/README.md)
├── notebooks/       Interactive development, step by step
│   └── 01_data_inventory.ipynb   <- START HERE (ready to run)
├── src/             Reusable modules (built together, one step at a time)
├── api/             FastAPI inference backend (built later)
├── dashboard/       Streamlit analyst dashboard (built later)
├── db/              SQLite database + notes
├── models/          Saved model artifacts (gitignored)
└── reports/         Figures, evaluation outputs, report assets
```

## How to use this skeleton

This is an **empty scaffold**. Every module under `src/`, `api/`, and
`dashboard/` is a placeholder stub with a docstring describing what goes there.
We fill them in one at a time, and you run + confirm each step before moving on.

### Setup (one time)

```bash
# From inside the nids-capstone folder
python -m venv .venv
# Windows:
.venv\Scripts\activate
# then:
pip install -r requirements.txt
```

### Next step

Open `notebooks/01_data_inventory.ipynb` and run it. Paste the outputs back
into our chat so we can proceed to preprocessing.

## Evaluation metrics (planned)

Accuracy, Precision, Recall, F1, ROC-AUC, False Positive Rate, **per-class F1**
for every attack category, **inference latency**, and **SHAP overhead**
(measured separately from inference).
