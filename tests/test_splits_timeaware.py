"""Tests for time-aware splits."""

import json
from pathlib import Path

import pandas as pd
import pytest

from qalice_core.splits import temporal_holdout


@pytest.fixture
def temporal_df():
    """Create DataFrame with temporal structure."""
    return pd.DataFrame(
        {
            "month_idx": [0, 0, 1, 1, 2, 2],
            "month": ["jan", "jan", "feb", "feb", "mar", "mar"],
            "y": [0, 1, 0, 1, 0, 1],
            "feature1": [1, 2, 3, 4, 5, 6],
            "feature2": [10, 20, 30, 40, 50, 60],
        }
    )


def test_temporal_holdout_split(temporal_df, tmp_path, monkeypatch):
    """Test temporal holdout creates correct train/valid split."""
    # Change to tmp directory for test isolation
    monkeypatch.chdir(tmp_path)

    train_df, valid_df = temporal_holdout(temporal_df, valid_fraction=1 / 3)

    # With 3 unique months and valid_fraction=1/3, expect 1 month for validation
    assert len(train_df) == 4  # months 0,1 (4 rows)
    assert len(valid_df) == 2  # month 2 (2 rows)

    # Validation should only contain the last month
    assert all(valid_df["month_idx"] == 2)

    # Training should contain earlier months
    assert set(train_df["month_idx"]) == {0, 1}


def test_splits_json_created(temporal_df, tmp_path, monkeypatch):
    """Test that splits.json is created with correct metadata."""
    # Change to tmp directory for test isolation
    monkeypatch.chdir(tmp_path)

    temporal_holdout(temporal_df, valid_fraction=1 / 3)

    # Check splits.json exists
    splits_path = Path("data/processed/bank_marketing/splits.json")
    assert splits_path.exists()

    # Check splits.json content
    with open(splits_path) as f:
        splits_info = json.load(f)

    assert splits_info["train_months"] == [0, 1]
    assert splits_info["valid_months"] == [2]
    assert splits_info["valid_fraction"] == 1 / 3
    assert splits_info["train_rows"] == 4
    assert splits_info["valid_rows"] == 2


def test_different_valid_fractions(temporal_df, tmp_path, monkeypatch):
    """Test different validation fractions."""
    monkeypatch.chdir(tmp_path)

    # Test with valid_fraction=0.5 (should give 2 months for validation)
    train_df, valid_df = temporal_holdout(temporal_df, valid_fraction=0.5)

    assert len(train_df) == 2  # month 0 only
    assert len(valid_df) == 4  # months 1,2
    assert set(valid_df["month_idx"]) == {1, 2}
    assert set(train_df["month_idx"]) == {0}
