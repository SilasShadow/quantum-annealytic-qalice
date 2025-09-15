"""Tests for feature toggle views."""

import pandas as pd
import pytest

from qalice_core.feature_engineering import build_feature_view


@pytest.fixture
def synthetic_df():
    """Create synthetic DataFrame with enriched schema."""
    return pd.DataFrame(
        {
            # Client data
            "age": [25, 30, 35, 40, 45, 50],
            "job": ["admin", "technician", "management", "admin", "technician", "management"],
            "marital": ["single", "married", "single", "married", "single", "married"],
            "education": [
                "university.degree",
                "high.school",
                "university.degree",
                "high.school",
                "university.degree",
                "high.school",
            ],
            "default": ["no", "no", "no", "no", "no", "no"],
            "housing": ["yes", "no", "yes", "no", "yes", "no"],
            "loan": ["no", "no", "yes", "no", "no", "yes"],
            # Contact data
            "contact": ["cellular", "telephone", "cellular", "telephone", "cellular", "telephone"],
            "month": ["jan", "jan", "feb", "feb", "mar", "mar"],
            "day_of_week": ["mon", "tue", "wed", "thu", "fri", "mon"],
            # Campaign data
            "campaign": [1, 2, 1, 3, 2, 1],
            "pdays": [999, 999, 5, 999, 10, 999],
            "previous": [0, 0, 1, 0, 2, 0],
            "poutcome": [
                "nonexistent",
                "nonexistent",
                "success",
                "nonexistent",
                "failure",
                "nonexistent",
            ],
            # Macro data
            "emp_var_rate": [1.1, 1.1, 1.4, 1.4, -1.8, -1.8],
            "cons_price_idx": [93.994, 93.994, 93.918, 93.918, 94.465, 94.465],
            "cons_conf_idx": [-36.4, -36.4, -42.7, -42.7, -41.8, -41.8],
            "euribor3m": [4.857, 4.857, 4.961, 4.961, 1.299, 1.299],
            "nr_employed": [5191.0, 5191.0, 5228.1, 5228.1, 5099.1, 5099.1],
            # Target and time
            "y": [0, 1, 0, 1, 0, 1],
            "month_idx": [0, 0, 1, 1, 2, 2],
        }
    )


def test_sentiment_toggle_features(synthetic_df):
    """Test that sentiment toggle correctly includes/excludes cons_conf_idx."""
    # Build with sentiment
    fv_with, schema_with, _ = build_feature_view(synthetic_df, include_sentiment=True)

    # Build without sentiment
    fv_without, schema_without, _ = build_feature_view(synthetic_df, include_sentiment=False)

    # Assert sentiment feature inclusion
    assert "cons_conf_idx" in schema_with["numeric_features"]
    assert "cons_conf_idx" not in schema_without["numeric_features"]

    # Assert scaled sentiment feature exists in with-view
    assert "cons_conf_idx_scaled" in fv_with.columns
    assert "cons_conf_idx_scaled" not in fv_without.columns


def test_passthrough_columns(synthetic_df):
    """Test that passthrough columns are present in both views."""
    fv_with, _, _ = build_feature_view(synthetic_df, include_sentiment=True)
    fv_without, _, _ = build_feature_view(synthetic_df, include_sentiment=False)

    passthrough_cols = ["y", "month", "month_idx", "cons_conf_idx"]

    for col in passthrough_cols:
        assert col in fv_with.columns
        assert col in fv_without.columns


def test_duration_excluded(synthetic_df):
    """Test that duration column is not present in feature views."""
    # Add duration to test exclusion
    synthetic_df["duration"] = [100, 200, 150, 300, 250, 180]

    fv_with, _, _ = build_feature_view(synthetic_df, include_sentiment=True)
    fv_without, _, _ = build_feature_view(synthetic_df, include_sentiment=False)

    assert "duration" not in fv_with.columns
    assert "duration" not in fv_without.columns


def test_one_hot_encoding(synthetic_df):
    """Test that categorical variables are one-hot encoded."""
    fv_with, schema_with, _ = build_feature_view(synthetic_df, include_sentiment=True)

    # Check that one-hot features are created
    assert len(schema_with["one_hot_features"]) > 0

    # Check specific one-hot columns exist
    assert any("job_" in col for col in fv_with.columns)
    assert any("marital_" in col for col in fv_with.columns)
