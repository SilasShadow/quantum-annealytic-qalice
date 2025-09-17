"""Minimal tests for baselines clustering module."""

import json
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from qalice_core.baselines import run_clustering


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
    numeric_cols = ["age_scaled", "campaign_scaled", "pdays_scaled"]
    
    # Create DataFrame
    df = pd.DataFrame(
        np.hstack([one_hot_data, numeric_data]),
        columns=one_hot_cols + numeric_cols
    )
    
    # Add required columns
    df["y"] = np.random.binomial(1, 0.1, n_rows)  # Bernoulli(0.1)
    df["month_idx"] = np.random.choice([0, 1, 2, 3, 4], n_rows)  # Blocks 0-4
    df["month"] = "jan"
    df["cons.conf.idx"] = -40.0
    
    return df


@pytest.fixture
def synthetic_schema():
    """Create matching schema for synthetic data."""
    return {
        "one_hot_features": [f"job_{i}" for i in range(5)] + [f"marital_{i}" for i in range(3)] + [f"education_{i}" for i in range(2)],
        "scaler_features": ["age_scaled", "campaign_scaled", "pdays_scaled"],
        "numeric_features": ["age_scaled", "campaign_scaled", "pdays_scaled"]
    }


@pytest.fixture
def synthetic_splits():
    """Create splits with last bucket as validation."""
    return {
        "train_months": [0, 1, 2, 3],
        "valid_months": [4]
    }


def test_run_clustering_minimal(tmp_path, synthetic_feature_view, synthetic_schema, synthetic_splits, monkeypatch):
    """Test baselines.run_clustering with synthetic data."""
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
    
    # Create interim data for KPI table
    interim_dir = tmp_path / "data" / "interim"
    interim_dir.mkdir(parents=True)
    interim_path = interim_dir / "bank_marketing_clean.csv"
    
    # Add required columns for KPI table
    interim_df = synthetic_feature_view.copy()
    interim_df["job"] = "admin"
    interim_df["marital"] = "single"
    interim_df["education"] = "university.degree"
    interim_df["age"] = 30
    interim_df["campaign"] = 1
    interim_df["pdays"] = 999
    interim_df["previous"] = 0
    interim_df.to_csv(interim_path, index=False)
    
    # Monkeypatch paths
    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        
        # Run clustering
        result_df = run_clustering(prefix="with_sentiment", k_list=[3])
    
    # Assert outputs exist
    reports_dir = tmp_path / "reports" / "baselines" / "with_sentiment"
    assert (reports_dir / "kmeans_k3_metrics.json").exists()
    assert (reports_dir / "kmeans_k3_kpi_valid.csv").exists()
    assert (reports_dir / "summary.csv").exists()
    
    # Load and check metrics
    with open(reports_dir / "kmeans_k3_metrics.json") as f:
        metrics = json.load(f)
    
    # Assert silhouette is not None if n_valid >= 100
    n_valid = (synthetic_feature_view["month_idx"] == 4).sum()
    if n_valid >= 100:
        assert metrics["silhouette"] is not None
    
    # Check result DataFrame
    assert len(result_df) == 2  # kmeans + gmm
    assert "silhouette" in result_df.columns