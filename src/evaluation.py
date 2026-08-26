"""Evaluation metrics.

Report (all on the held-out 20% test set):
    Accuracy, Precision, Recall, F1, ROC-AUC, False Positive Rate,
    per-class F1 for every CICIDS2017 attack category,
    inference latency (ms),
    and SHAP overhead measured SEPARATELY from inference latency (Gap 5).
"""

from __future__ import annotations


def classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Compute the full metric suite, including per-class F1 and FPR."""
    raise NotImplementedError("Implemented during the evaluation step.")


def measure_inference_latency(model, X) -> float:
    """Return mean per-sample inference latency in milliseconds."""
    raise NotImplementedError("Implemented during the evaluation step.")
