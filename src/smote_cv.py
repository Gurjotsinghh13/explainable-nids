"""Leakage-free, within-fold SMOTE for the stacking ensemble (core novelty).

Applying SMOTE to the whole training set before cross-validation lets
synthetic points interpolated from validation-fold samples enter the training
folds. In a stacking ensemble this contaminates the out-of-fold predictions
that train the meta-learner. Putting SMOTE *inside* each base learner's
imbalanced-learn Pipeline guarantees it only ever resamples the training part
of a fold, and it is skipped automatically at prediction time.

``SmoteFloor`` is a small picklable class (a plain closure cannot be pickled,
which breaks ``joblib.dump`` of the trained model).
"""

from __future__ import annotations

from collections import Counter

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from .config import RANDOM_STATE, SMOTE_FLOOR, SMOTE_K_NEIGHBORS


class SmoteFloor:
    """Sampling strategy: oversample every class below ``floor`` up to ``floor``.

    Classes at or above the floor are left untouched; nothing is undersampled.
    The target is computed from the class counts of the data passed in, i.e.
    from each training fold separately.
    """

    def __init__(self, floor: int = SMOTE_FLOOR):
        self.floor = floor

    def __call__(self, y):
        counts = Counter(y)
        return {cls: self.floor for cls, n in counts.items() if n < self.floor}

    def __repr__(self):
        return f"SmoteFloor(floor={self.floor})"


def make_within_fold_pipeline(estimator, floor: int = SMOTE_FLOOR,
                              k_neighbors: int = SMOTE_K_NEIGHBORS,
                              name: str = "clf") -> ImbPipeline:
    """Wrap ``estimator`` so SMOTE runs only on each fold's training data."""
    return ImbPipeline([
        ("smote", SMOTE(sampling_strategy=SmoteFloor(floor),
                        k_neighbors=k_neighbors, random_state=RANDOM_STATE)),
        (name, estimator),
    ])
