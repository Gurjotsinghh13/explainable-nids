"""Preprocessing for CICIDS2017.

Two entry points:
    preprocess_training(data)  -> reproducible training pipeline
    preprocess_inference(...)  -> transform live flows for the FastAPI backend

Pipeline (in order):
    1. Merge Web Attack subtypes; drop Heartbleed and Infiltration (11 classes).
    2. Remove rows with infinite or NaN values.
    3. Drop constant and duplicate columns.
    4. Drop socket-identifying (leakage) features.
    5. Stratified 80/20 split; the test set is held out from here on.
    6. Min-Max scale to [0, 1] with the scaler fitted on TRAIN only.
    7. Label-encode the target.

SMOTE is deliberately NOT applied here. It runs inside each cross-validation
fold of the stacking ensemble (see smote_cv.py) so that no synthetic sample
reaches the meta-learner's training data.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from .config import (DROPPED_CLASSES, LABEL_COLUMN, LEAKAGE_FEATURES,
                     RANDOM_STATE, TEST_SIZE)


def find_constant_columns(frame: pd.DataFrame) -> list[str]:
    """Columns with a single unique value (zero variance)."""
    nun = frame.nunique(dropna=False)
    return nun[nun <= 1].index.tolist()


def find_duplicate_columns(frame: pd.DataFrame) -> list[str]:
    """Columns whose values exactly duplicate an earlier column."""
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
    """Merge the three Web Attack subtypes and drop the ultra-rare classes."""
    df = df.copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()
    df.loc[df[LABEL_COLUMN].str.startswith("Web Attack"), LABEL_COLUMN] = "Web Attack"
    return df[~df[LABEL_COLUMN].isin(DROPPED_CLASSES)].reset_index(drop=True)


def preprocess_training(data: pd.DataFrame, verbose: bool = True) -> dict:
    """Run the full training preprocessing pipeline.

    Returns a dict with X_train, X_test (float32), y_train, y_test (encoded),
    feature_names, label_classes, scaler and label_encoder.
    """
    df = clean_labels(data)

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if verbose:
        print(f"Removed {before - len(df)} rows with inf/NaN")

    features = df.drop(columns=[LABEL_COLUMN])
    to_drop = set(find_constant_columns(features)) | set(find_duplicate_columns(features))
    to_drop |= {c for c in LEAKAGE_FEATURES if c in df.columns}
    df = df.drop(columns=sorted(to_drop))
    if verbose:
        print(f"Dropped {len(to_drop)} columns -> {df.shape[1] - 1} features")

    X = df.drop(columns=[LABEL_COLUMN])
    y = df[LABEL_COLUMN]
    feature_names = X.columns.tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_te_enc = le.transform(y_te)

    return {"X_train": X_tr, "X_test": X_te,
            "y_train": y_tr_enc, "y_test": y_te_enc,
            "feature_names": feature_names,
            "label_classes": le.classes_.tolist(),
            "scaler": scaler, "label_encoder": le}


def preprocess_inference(df_raw: pd.DataFrame, scaler, feature_order: list[str]) -> np.ndarray:
    """Transform raw flow(s) into a model-ready array using the SAVED scaler.

    Robust to messy input: missing columns are filled with 0, infinities and
    NaNs are neutralised, extra columns are ignored, and scaled values are
    clipped to the [0, 1] range the model was trained on.
    """
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    df = df.reindex(columns=feature_order)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = scaler.transform(df.values).astype(np.float32)
    return np.clip(X, 0.0, 1.0)
