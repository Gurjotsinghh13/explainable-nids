"""Central configuration: paths and constants.

Keep every hard-coded path and magic number here so the rest of the code
stays clean and reproducible.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Default location of the extracted CICIDS2017 (MachineLearningCVE) CSVs.
# Change this if you move the dataset into the project's data/ folder.
DATA_DIR = Path(r"D:\Capstone Project\MachineLearningCVE")

# Project-relative paths (resolved from this file's location).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DB_PATH = PROJECT_ROOT / "db" / "nids.db"

# --------------------------------------------------------------------------- #
# Data / preprocessing constants
# --------------------------------------------------------------------------- #
LABEL_COLUMN = "Label"           # after stripping whitespace from column names
RANDOM_STATE = 42
TEST_SIZE = 0.20                 # stratified 80/20 hold-out
CV_FOLDS = 5                     # cross-validation folds for stacking / SMOTE
SMOTE_K_NEIGHBORS = 5

# Features that identify the socket/flow and would leak label information.
# NOTE: the MachineLearningCVE version already omits source/dest IP; verify the
# exact present columns during the data-inventory step before dropping.
LEAKAGE_CANDIDATE_FEATURES = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
]

# --------------------------------------------------------------------------- #
# Model constants (finalised when we build src/models.py)
# --------------------------------------------------------------------------- #
# Base learners: RandomForest + XGBoost.  Meta-learner: LogisticRegression.
# Hyperparameters are decided together during the modelling step.
