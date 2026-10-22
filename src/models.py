"""Model definitions: baselines and the RF + XGBoost -> LR stacking ensemble."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .config import (CV_FOLDS, META_PARAMS, RANDOM_STATE, RF_PARAMS,
                     SMOTE_FLOOR, XGB_PARAMS)
from .smote_cv import make_within_fold_pipeline


def build_random_forest(n_jobs: int = 4) -> RandomForestClassifier:
    """Random Forest (100 trees, max depth 30, min leaf 2).

    ``n_jobs`` is capped because unbounded parallel RF training on 2.26M rows
    exhausted RAM and overheated the development laptop.
    """
    return RandomForestClassifier(n_jobs=n_jobs, **RF_PARAMS)


def build_xgboost(device: str = "cpu") -> XGBClassifier:
    """XGBoost (100 estimators, max depth 8, learning rate 0.3).

    Pass ``device="cuda"`` to train on an NVIDIA GPU.
    """
    return XGBClassifier(device=device, **XGB_PARAMS)


def build_stacking_ensemble(floor: int = SMOTE_FLOOR, n_jobs: int = 4,
                            device: str = "cpu", cv_folds: int = CV_FOLDS) -> StackingClassifier:
    """RF + XGBoost base learners (each with within-fold SMOTE) -> LR meta-learner.

    ``StackingClassifier`` builds the meta-learner's training data from
    out-of-fold predictions. Because SMOTE sits inside each base pipeline, it
    only resamples the training part of every fold; held-out folds are
    predicted on real, untouched flows.
    """
    rf = make_within_fold_pipeline(build_random_forest(n_jobs), floor, name="rf")
    xgb = make_within_fold_pipeline(build_xgboost(device), floor, name="xgb")
    return StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=LogisticRegression(**META_PARAMS),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=1,          # sequential: keeps memory bounded on a 16 GB machine
        passthrough=False,
    )


def get_xgb_base_learner(stack: StackingClassifier):
    """Return the fitted XGBoost base learner from a trained stack (used for SHAP)."""
    return stack.named_estimators_["xgb"].named_steps["xgb"]
