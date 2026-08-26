"""Load and combine the 8 CICIDS2017 (MachineLearningCVE) CSV files.

Built together in the data-inventory / preprocessing step. Stubs below define
the intended interface; they are not implemented yet.
"""

from __future__ import annotations

import pandas as pd


def load_all_csvs(data_dir=None) -> pd.DataFrame:
    """Read all 8 CSVs, strip whitespace from column names, and concatenate.

    Args:
        data_dir: Directory holding the CSVs. Defaults to config.DATA_DIR.

    Returns:
        A single combined DataFrame with a cleaned ``Label`` column.
    """
    raise NotImplementedError("Implemented during the data-loading step.")
