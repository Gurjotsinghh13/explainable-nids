"""Build the balanced replay stream used by the Streamlit dashboard.

Takes up to 150 RAW (unscaled) flows per class so the demo shows every attack
type, not just benign traffic.  Run after training:
    python -m scripts.make_replay_sample
"""

from __future__ import annotations

import joblib
import numpy as np

from src.config import LABEL_COLUMN, PREPROCESSED_FILE, REPLAY_FILE, RANDOM_STATE
from src.data_loading import load_all_csvs
from src.preprocessing import clean_labels


def main(per_class: int = 150):
    feats = joblib.load(PREPROCESSED_FILE)["feature_names"]
    data = clean_labels(load_all_csvs())
    data = data[feats + [LABEL_COLUMN]].replace([np.inf, -np.inf], np.nan).dropna()
    sample = (data.groupby(LABEL_COLUMN, group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), per_class), random_state=RANDOM_STATE))
                  .sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True))
    sample.to_csv(REPLAY_FILE, index=False)
    print(f"Saved {len(sample)} flows to {REPLAY_FILE}")
    print(sample[LABEL_COLUMN].value_counts())


if __name__ == "__main__":
    main()
