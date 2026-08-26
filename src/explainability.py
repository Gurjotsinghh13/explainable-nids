"""SHAP explainability using TreeExplainer.

Outputs:
    - Global feature-importance summary across the test set.
    - Per-alert SHAP waterfall for an individual flagged flow (top contributing
      features + direction), surfaced to the analyst in the dashboard.

SHAP computation time is measured separately from model inference latency so we
can quantify SHAP overhead inside the stacking pipeline (research Gap 5).
"""

from __future__ import annotations


def build_explainer(model):
    """Return a shap.TreeExplainer for a tree-based (base) model."""
    raise NotImplementedError("Implemented during the explainability step.")


def global_summary(explainer, X):
    """Produce global SHAP feature-importance values / summary plot."""
    raise NotImplementedError("Implemented during the explainability step.")


def explain_one(explainer, x_row):
    """Produce a per-alert SHAP explanation (waterfall) for a single flow."""
    raise NotImplementedError("Implemented during the explainability step.")
