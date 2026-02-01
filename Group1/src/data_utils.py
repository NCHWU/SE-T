"""
Data loading utilities for the bias detection pipeline.

This module provides shared data loading functions used by both training
and testing modules, eliminating code duplication.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from constants import LABEL_COLUMN, LEAKED_COLUMNS


def load_data(
    data_path: str | Path,
    label_column: str = LABEL_COLUMN,
    return_raw: bool = False,
) -> Tuple[pd.DataFrame, pd.Series | None] | Tuple[pd.DataFrame, pd.Series | None, pd.DataFrame]:
    """
    Load dataset and prepare features and labels.

    Parameters
    ----------
    data_path : str or Path
        Path to the CSV file.
    label_column : str, default="checked"
        Name of the target column.
    return_raw : bool, default=False
        If True, also return the raw dataframe before processing.
        Useful for partition tests that need original column values.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with label and leaked columns dropped.
        All values cast to float32 for ONNX compatibility.
    y : pd.Series or None
        Target labels if label_column exists, None otherwise.
    df_raw : pd.DataFrame (only if return_raw=True)
        Original dataframe before any processing.

    Examples
    --------
    Basic usage for training:

    >>> X, y = load_data("data/train.csv")

    With raw dataframe for partition tests:

    >>> X, y, df_raw = load_data("data/test.csv", return_raw=True)
    """
    df = pd.read_csv(data_path)

    # Extract labels if present
    y = df[label_column] if label_column in df.columns else None

    # Drop label and leaked columns
    columns_to_drop = [label_column] + LEAKED_COLUMNS
    columns_to_drop = [c for c in columns_to_drop if c in df.columns]
    X = df.drop(columns=columns_to_drop).astype(np.float32)

    if return_raw:
        return X, y, df
    return X, y
