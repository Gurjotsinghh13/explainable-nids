"""Evaluation: metrics, per-class FPR, latency, SHAP overhead, CV stability."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .config import CV_FOLDS, CV_SMOTE_FLOOR, CV_SUBSAMPLE, RANDOM_STATE


def per_class_fpr(cm: np.ndarray) -> np.ndarray:
    """False-positive rate per class from a multi-class confusion matrix."""
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    return fp / (fp + tn)


def classification_metrics(y_true, y_pred, class_names, y_proba=None) -> dict:
    """Headline metrics plus a per-class table (precision, recall, F1, FPR)."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fpr = per_class_fpr(cm)
    rep = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                target_names=class_names, output_dict=True,
                                zero_division=0)
    per_class = pd.DataFrame({
        "precision": [rep[c]["precision"] for c in class_names],
        "recall": [rep[c]["recall"] for c in class_names],
        "f1": [rep[c]["f1-score"] for c in class_names],
        "FPR": fpr,
        "support": [int(rep[c]["support"]) for c in class_names],
    }, index=class_names)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_fpr": float(fpr.mean()),
        "confusion_matrix": cm,
        "per_class": per_class,
    }
    if y_proba is not None:
        out["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    return out


def class_recall(y_true, y_pred, class_index: int) -> float:
    """Recall of a single class (used for the Bot-recall comparison)."""
    return recall_score(y_true, y_pred, labels=[class_index], average="macro")


def measure_inference_latency(model, x_one, repeats: int = 100) -> float:
    """Mean milliseconds to classify one flow."""
    t0 = time.perf_counter()
    for _ in range(repeats):
        model.predict(x_one)
    return (time.perf_counter() - t0) / repeats * 1000


def measure_shap_overhead(explainer, x_one, repeats: int = 100) -> float:
    """Mean milliseconds to compute a SHAP explanation for one flow."""
    t0 = time.perf_counter()
    for _ in range(repeats):
        explainer(x_one)
    return (time.perf_counter() - t0) / repeats * 1000


def cross_validated_stability(build_fn, X, y, subsample: int = CV_SUBSAMPLE,
                              floor: int = CV_SMOTE_FLOOR, folds: int = CV_FOLDS) -> dict:
    """5-fold stratified CV of the stacking pipeline on a stratified subsample.

    Uses a lighter SMOTE floor than the full model so the run fits on a
    laptop; it measures stability (mean +/- std), not headline performance.
    """
    if subsample and subsample < len(y):
        X, _, y, _ = train_test_split(X, y, train_size=subsample, stratify=y,
                                      random_state=RANDOM_STATE)
    res = cross_validate(build_fn(floor=floor), X, y,
                         cv=StratifiedKFold(folds, shuffle=True, random_state=RANDOM_STATE),
                         scoring=["accuracy", "f1_macro", "f1_weighted"], n_jobs=1)
    return {m: (float(res[f"test_{m}"].mean()), float(res[f"test_{m}"].std()))
            for m in ["accuracy", "f1_macro", "f1_weighted"]}
