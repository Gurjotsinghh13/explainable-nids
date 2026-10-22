"""SHAP explainability with TreeExplainer on the XGBoost base learner.

TreeExplainer computes exact Shapley values for tree models, but the
stack's final estimator is Logistic Regression. Explanations are therefore
computed on the XGBoost base learner, which consumes the 64 raw flow
features, so each alert is attributed to interpretable network features.
"""

from __future__ import annotations

import numpy as np
import shap

from .models import get_xgb_base_learner


def build_explainer(stack) -> shap.TreeExplainer:
    """TreeExplainer on the fitted XGBoost base learner (run on CPU)."""
    xgb = get_xgb_base_learner(stack)
    try:
        xgb.set_params(device="cpu")
    except Exception:
        pass
    return shap.TreeExplainer(xgb)


def global_importance(explainer, X_sample, feature_names, top_k: int = 15):
    """Mean |SHAP| per feature (averaged over classes), sorted descending."""
    sv = np.asarray(explainer.shap_values(X_sample))      # (n, features, classes)
    imp = np.abs(sv).mean(axis=0)
    if imp.ndim == 2:
        imp = imp.sum(axis=1)
    order = np.argsort(imp)[::-1][:top_k]
    return [(feature_names[i], float(imp[i])) for i in order]


def explain_one(explainer, x_row, class_index: int, feature_names, top_k: int = 8):
    """Top contributing features for one flow toward ``class_index``."""
    exp = explainer(x_row)
    vals = np.asarray(exp.values)[0, :, class_index]
    order = np.argsort(np.abs(vals))[::-1][:top_k]
    return [{"feature": feature_names[i],
             "shap_value": round(float(vals[i]), 4),
             "direction": "increases" if vals[i] > 0 else "decreases"}
            for i in order]
