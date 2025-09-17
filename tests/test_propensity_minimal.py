"""Minimal tests for propensity modeling module."""

import json
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from qalice_core.propensity import train_and_evaluate


@pytest.fixture
def synthetic_feature_view():
    """Create tiny synthetic feature view with 200 rows."""
    np.random.seed(42)
    n_rows = 200
    
    # Binary one-hot features (10 columns)
    one_hot_data = np.random.binomial(1, 0.3, (n_rows, 10))
    one_hot_cols = [f"job_{i}" for i in range(5)] + [f"marital_{i}" for i in range(3)] + [f"education_{i}" for i in range(2)]
    
    # Numeric features (3 columns)
    numeric_data = np.random.randn(n_rows, 3)
    numeric_cols = ["age", "campaign", "pdays"]
    
    # Create DataFrame
    df = pd.DataFrame(
        np.hstack([one_hot_data, numeric_data]),
        columns=one_hot_cols + numeric_cols
    )
    
    # Add required columns
    df["y"] = np.random.binomial(1, 0.1, n_rows)  # Bernoulli(0.1)
    df["month_idx"] = np.random.choice([0, 1, 2, 3, 4], n_rows)  # Blocks 0-4
    
    return df


@pytest.fixture
def synthetic_schema():
    """Create matching schema for synthetic data."""
    return {
        "one_hot_features": [f"job_{i}" for i in range(5)] + [f"marital_{i}" for i in range(3)] + [f"education_{i}" for i in range(2)],
        "numeric_features": ["age", "campaign", "pdays"]
    }


@pytest.fixture
def synthetic_splits():
    """Create splits with last bucket as validation."""
    return {
        "train_months": [0, 1, 2, 3],
        "valid_months": [4]
    }


def test_train_and_evaluate_minimal(tmp_path, synthetic_feature_view, synthetic_schema, synthetic_splits, monkeypatch):
    """Test propensity.train_and_evaluate with synthetic data."""
    # Setup tmp paths
    data_dir = tmp_path / "data" / "processed" / "bank_marketing"
    data_dir.mkdir(parents=True)
    
    # Write synthetic data
    feature_path = data_dir / "with_sentiment_feature_view.parquet"
    synthetic_feature_view.to_parquet(feature_path)
    
    schema_path = data_dir / "with_sentiment_feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(synthetic_schema, f)
    
    splits_path = data_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(synthetic_splits, f)
    
    # Monkeypatch paths
    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        
        # Run propensity modeling
        metrics = train_and_evaluate("with_sentiment")
    
    # Assert outputs exist
    reports_dir = tmp_path / "reports" / "propensity" / "with_sentiment"
    assert (reports_dir / "metrics.json").exists()
    assert (reports_dir / "model.pkl").exists()
    assert (reports_dir / "decile_lift.csv").exists()
    
    # Load and check metrics
    with open(reports_dir / "metrics.json") as f:
        saved_metrics = json.load(f)
    
    # Assert required metric keys exist
    required_keys = ["auc", "pr_auc", "logloss", "brier"]
    for key in required_keys:
        assert key in saved_metrics
        assert saved_metrics[key] is not None
    
    # Check decile lift table
    decile_df = pd.read_csv(reports_dir / "decile_lift.csv")
    assert len(decile_df) == 10  # Should have 10 deciles
    assert "decile" in decile_df.columns
    assert "conversion_rate" in decile_df.columns
    assert "lift_vs_overall" in decile_df.columns
    
    # Check returned metrics match saved metrics
    assert metrics["auc"] == saved_metrics["auc"]