"""NIDS capstone source package.

Modules are filled in interactively, one step at a time:
    config          -> paths and constants
    data_loading    -> load and combine the 8 CICIDS2017 CSVs
    preprocessing   -> cleaning, feature handling, encoding, split
    smote_cv        -> within-fold SMOTE (core novelty)
    models          -> RF + XGBoost base learners, LR meta-learner, stacking
    evaluation      -> metrics incl. per-class F1, FPR, latency, SHAP overhead
    explainability  -> SHAP TreeExplainer (global + per-alert)
"""
