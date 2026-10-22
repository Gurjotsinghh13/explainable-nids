"""Central configuration: paths and constants used across the project."""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Location of the extracted CICIDS2017 (MachineLearningCVE) CSV files.
# Change this to wherever the 8 CSVs live on your machine.
DATA_DIR = Path(r"D:\Capstone Project\MachineLearningCVE")

# NIDS_ROOT lets tests point artefacts at a temporary folder.
PROJECT_ROOT = Path(os.environ.get("NIDS_ROOT", Path(__file__).resolve().parents[1]))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DB_PATH = PROJECT_ROOT / "db" / "nids.db"

PREPROCESSED_FILE = PROCESSED_DIR / "preprocessed.joblib"
REPLAY_FILE = PROCESSED_DIR / "replay_sample.csv"
MODEL_FILE = MODELS_DIR / "stacking_model.joblib"
SCALER_FILE = MODELS_DIR / "scaler.joblib"
ENCODER_FILE = MODELS_DIR / "label_encoder.joblib"

# --------------------------------------------------------------------------- #
# Data / preprocessing
# --------------------------------------------------------------------------- #
LABEL_COLUMN = "Label"
RANDOM_STATE = 42
TEST_SIZE = 0.20                       # stratified 80/20 hold-out

# Socket / flow identifiers that would leak the label. Only those present in
# the loaded data are dropped (MachineLearningCVE exposes Destination Port).
LEAKAGE_FEATURES = ["Flow ID", "Source IP", "Source Port",
                    "Destination IP", "Destination Port", "Timestamp"]

# Classes too small to train or evaluate reliably (11 and 36 records).
DROPPED_CLASSES = ["Heartbleed", "Infiltration"]

# --------------------------------------------------------------------------- #
# Model / training
# --------------------------------------------------------------------------- #
CV_FOLDS = 5                           # folds for stacking and for CV evaluation
SMOTE_K_NEIGHBORS = 5
SMOTE_FLOOR = 50_000                   # classes below this are oversampled up to it

RF_PARAMS = dict(n_estimators=100, max_depth=30, min_samples_leaf=2,
                 random_state=RANDOM_STATE)
XGB_PARAMS = dict(n_estimators=100, max_depth=8, learning_rate=0.3,
                  tree_method="hist", eval_metric="mlogloss",
                  random_state=RANDOM_STATE)
META_PARAMS = dict(max_iter=1000)

# Cross-validation stability run (lighter, to fit on a laptop)
CV_SUBSAMPLE = 300_000
CV_SMOTE_FLOOR = 15_000
