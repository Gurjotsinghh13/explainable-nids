"""Load and combine the 8 CICIDS2017 (MachineLearningCVE) CSV files."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

from .config import DATA_DIR


def load_all_csvs(data_dir: str | Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """Read every CSV in ``data_dir``, strip column-name whitespace and concatenate.

    The CICIDS2017 files have leading spaces in most column names
    (e.g. ``' Label'``); stripping them gives stable names such as ``Label``.
    """
    data_dir = Path(data_dir or DATA_DIR)
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False, encoding="latin-1")
        df.columns = df.columns.str.strip()
        frames.append(df)
        if verbose:
            print(f"{os.path.basename(f):60s} {df.shape}")
    data = pd.concat(frames, ignore_index=True)
    if verbose:
        print("Combined:", data.shape)
    return data
