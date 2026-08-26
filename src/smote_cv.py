"""Leakage-free, within-fold SMOTE for the stacking pipeline (CORE NOVELTY).

Global SMOTE (resampling the whole training set before cross-validation) leaks
synthetic minority samples from a fold into its own validation split, which
contaminates the meta-learner's training data. The correct approach applies
SMOTE independently inside each CV fold.

The clean way to guarantee this in scikit-learn is imblearn's Pipeline, which
only resamples the training portion of each fold:

    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import SMOTE
    pipe = Pipeline([("smote", SMOTE(...)), ("clf", base_learner)])

We build and verify this together in the stacking step.
"""

from __future__ import annotations


def make_within_fold_pipeline(base_estimator):
    """Wrap a base estimator so SMOTE runs only on each fold's training data."""
    raise NotImplementedError("Implemented during the stacking-ensemble step.")
