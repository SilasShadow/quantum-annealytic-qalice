"""Time-aware train/validation splits using month_idx."""

import json
import logging
from math import ceil
from pathlib import Path
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def temporal_holdout(
    df: pd.DataFrame, valid_fraction: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform time-aware train/validation split using month_idx.

    Args:
        df: DataFrame with month_idx column
        valid_fraction: Fraction of months to use for validation

    Returns:
        Tuple of (train_df, valid_df)
    """
    # Get unique months in order
    unique_months = sorted(df["month_idx"].unique())

    # Calculate validation months
    n_valid = ceil(valid_fraction * len(unique_months))
    valid_months = unique_months[-n_valid:]
    train_months = unique_months[:-n_valid]

    # Split data
    train_df = df[df["month_idx"].isin(train_months)].copy()
    valid_df = df[df["month_idx"].isin(valid_months)].copy()

    # Persist split metadata
    output_dir = Path("data/processed/bank_marketing")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_info = {
        "train_months": [int(m) for m in train_months],
        "valid_months": [int(m) for m in valid_months],
        "valid_fraction": valid_fraction,
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
    }

    splits_path = output_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits_info, f, indent=2)

    logger.info(f"Temporal split: {len(train_df)} train, {len(valid_df)} valid rows")
    logger.info(f"Train months: {train_months}, Valid months: {valid_months}")
    logger.info(f"Split metadata saved to {splits_path}")

    return train_df, valid_df
