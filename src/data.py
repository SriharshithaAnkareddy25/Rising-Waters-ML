from __future__ import annotations

import pandas as pd

from .config import ALL_FEATURES, DATA_PATH, MODEL_FEATURES, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load and validate the immutable source dataset."""
    frame = pd.read_excel(path)
    expected = ALL_FEATURES + [TARGET]
    missing = sorted(set(expected) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(expected))
    if missing or unexpected:
        raise ValueError(f"Dataset schema mismatch; missing={missing}, unexpected={unexpected}")
    if frame[expected].isna().any().any():
        raise ValueError("Dataset contains missing values")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must be binary with values 0 and 1")
    return frame[expected].copy()


def model_xy(frame: pd.DataFrame):
    return frame[MODEL_FEATURES].copy(), frame[TARGET].astype(int).copy()
