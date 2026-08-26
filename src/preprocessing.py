"""Preprocessing for the CICIDS2017 NIDS capstone.

Two entry points:
    preprocess_training(...)  -> full reproducible training pipeline
    preprocess_inference(...) -> transform live flows for the FastAPI backend

Methodology (locked):
    1. Remove rows with infinite or NaN values.
    2. Drop constant and duplicate columns.
    3. Remove socket-identifying (leakage) features.
    4. Min-Max scale to [0, 1]  (scaler fit on TRAIN only).
    5. Label-encode the multi-class target.
    6. Stratified 80/20 split; test set held out.

SMOTE is NOT applied here — it runs strictly within each CV fold during
stacking (see notebook 04), which is the project's core novelty.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
LABEL_COLUMN = "Label"

# Socket / flow identifiers that leak label information. Only those actually
# present in the loaded frame are dropped (MachineLearningCVE usually has just
# "Destination Port").
LEAKAGE_FEATURES = [
    "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port", "Timestamp",
]

# Classes removed from the study: too few samples to train or evaluate reliably.
DROPPED_CLASSES = ["Heartbleed", "Infiltration"]

RANDOM_STATE = 42
TEST_SIZE = 0.20


# --------------------------------------------------------------------------- #
# Column helpers (shared with the notebook logic)
# --------------------------------------------------------------------------- #
def find_constant_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns with a single unique value (zero variance)."""
    nun = frame.nunique(dropna=False)
    return nun[nun <= 1].index.tolist()


def find_duplicate_columns(frame: pd.DataFrame) -> list[str]:
    """Return columns whose values exactly duplicate an earlier column."""
    seen: dict[str, str] = {}
    dupes: list[str] = []
    for col in frame.columns:
        h = hashlib.md5(
            pd.util.hash_pandas_object(frame[col], index=False).values.tobytes()
        ).hexdigest()
        if h in seen and frame[col].equals(frame[seen[h]]):
            dupes.append(col)
        else:
            seen.setdefault(h, col)
    return dupes


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Merge Web Attack subtypes into one class and drop the ultra-rare classes."""
    df = df.copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].str.strip()
    df.loc[df[LABEL_COLUMN].str.startswith("Web Attack"), LABEL_COLUMN] = "Web Attack"
    df = df[~df[LABEL_COLUMN].isin(DROPPED_CLASSES)].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Training pipeline
# --------------------------------------------------------------------------- #
def preprocess_training(data: pd.DataFrame):
    """Run the full training preprocessing pipeline.

    Args:
        data: Combined raw DataFrame of all 8 CSVs (column names already
              stripped of whitespace) with a ``Label`` column.

    Returns:
        dict with X_train, X_test (float32), y_train, y_test (encoded),
        feature_names, label_classes, scaler, label_encoder.
    """
    df = clean_labels(data)

    # 1. remove infinite / NaN rows
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # 2. drop constant + duplicate feature columns
    features = df.drop(columns=[LABEL_COLUMN])
    to_drop = set(find_constant_columns(features)) | set(find_duplicate_columns(features))
    df = df.drop(columns=list(to_drop))

    # 3. drop leakage features that are present
    leakage_present = [c for c in LEAKAGE_FEATURES if c in df.columns]
    df = df.drop(columns=leakage_present)

    # 4. split X / y, stratified 80/20
    X = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN]
    feature_names = X.columns.tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 5. scale (fit on TRAIN only) -> float32
    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    # 6. label-encode
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_te_enc = le.transform(y_te)

    return {
        "X_train": X_tr, "X_test": X_te,
        "y_train": y_tr_enc, "y_test": y_te_enc,
        "feature_names": feature_names,
        "label_classes": le.classes_.tolist(),
        "scaler": scaler, "label_encoder": le,
    }


# --------------------------------------------------------------------------- #
# Inference pipeline (used by the FastAPI backend)
# --------------------------------------------------------------------------- #
def preprocess_inference(df_raw: pd.DataFrame, scaler, feature_order: list[str]) -> np.ndarray:
    """Transform one or more raw flows into a model-ready array.

    Applies the SAME feature selection and the SAVED training scaler so live
    predictions match the trained model. Robust to messy input: missing columns
    are created empty, infinities/NaNs are neutralised, and scaled values are
    clipped to the [0, 1] range the model was trained on.

    Args:
        df_raw:        Raw flow(s); column names may contain whitespace.
        scaler:        The fitted MinMaxScaler saved during training.
        feature_order: The exact 64 feature names, in training order.

    Returns:
        float32 ndarray of shape (n_rows, 64), ready for model.predict.
    """
    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    # select the exact training features, in order (missing -> NaN column)
    df = df.reindex(columns=feature_order)

    # neutralise inf / NaN, then coerce to numeric
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = scaler.transform(df.values).astype(np.float32)
    return np.clip(X, 0.0, 1.0)