"""Test runtime_sentiment visualization module."""

import os
import pandas as pd
import pytest

from qalice_viz.runtime_sentiment import build_runtime_bar, build_sentiment_toggle_summary


def test_build_runtime_bar():
    """Test runtime bar chart generation."""
    path = build_runtime_bar("with_sentiment")
    assert os.path.exists(path)
    assert path.endswith("runtime_bar_with_sentiment.png")


def test_build_sentiment_toggle_summary():
    """Test sentiment toggle summary generation."""
    df = build_sentiment_toggle_summary()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 2
    assert "prefix" in df.columns
    assert "algo" in df.columns
    assert "silhouette" in df.columns
    assert "propensity_auc" in df.columns
    assert "top_decile_lift" in df.columns
    
    # Check CSV was created
    assert os.path.exists("reports/figures/sentiment_toggle_summary.csv")