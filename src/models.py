"""Model definitions: RF + XGBoost base learners, LR meta-learner, stacking.

Architecture (locked):
    base learners : RandomForestClassifier, XGBClassifier
    meta-learner  : LogisticRegression
    strategy      : stacking on out-of-fold predictions, with within-fold SMOTE

Hyperparameters are chosen together during the modelling step.
"""

from __future__ import annotations


def build_random_forest():
    """Return a configured RandomForestClassifier."""
    raise NotImplementedError("Implemented during the modelling step.")


def build_xgboost():
    """Return a configured XGBClassifier (multi-class)."""
    raise NotImplementedError("Implemented during the modelling step.")


def build_stacking_ensemble():
    """Assemble the RF + XGB -> LR stacking ensemble with within-fold SMOTE."""
    raise NotImplementedError("Implemented during the stacking-ensemble step.")
