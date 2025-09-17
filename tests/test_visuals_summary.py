"""Minimal tests for visuals summary module."""

import json
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from qlattice_viz.visuals import build_comparison_summary


def test_build_comparison_summary_minimal(tmp_path, monkeypatch):
    """Test visuals.build_comparison_summary with minimal KPI data."""
    # Create minimal directory structure
    reports_dir = tmp_path / "reports"
    
    # Create baseline metrics for both prefixes
    for prefix in ["with_sentiment", "no_sentiment"]:
        baseline_dir = reports_dir / "baselines" / prefix
        baseline_dir.mkdir(parents=True)
        
        # Create minimal clustering metrics
        metrics = {
            "silhouette": 0.5,
            "calinski_harabasz": 100.0,
            "davies_bouldin": 1.2,
            "n_valid": 150
        }
        
        metrics_path = baseline_dir / "kmeans_k3_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        
        # Create minimal KPI CSV
        kpi_data = pd.DataFrame({
            "cluster_id": [0, 1, 2],
            "size": [50, 60, 40],
            "size_pct": [33.3, 40.0, 26.7],
            "conversion_rate": [0.1, 0.15, 0.08]
        })
        
        kpi_path = baseline_dir / "kmeans_k3_kpi_valid.csv"
        kpi_data.to_csv(kpi_path, index=False)
    
    # Create propensity metrics for both prefixes
    for prefix in ["with_sentiment", "no_sentiment"]:
        prop_dir = reports_dir / "propensity" / prefix
        prop_dir.mkdir(parents=True)
        
        # Create minimal propensity metrics
        prop_metrics = {
            "auc": 0.75,
            "pr_auc": 0.25,
            "brier": 0.08,
            "logloss": 0.35
        }
        
        prop_metrics_path = prop_dir / "metrics.json"
        with open(prop_metrics_path, "w") as f:
            json.dump(prop_metrics, f)
        
        # Create minimal decile lift
        decile_data = pd.DataFrame({
            "decile": range(1, 11),
            "n": [20] * 10,
            "conversion_rate": np.linspace(0.2, 0.05, 10),
            "lift_vs_overall": np.linspace(2.0, 0.5, 10)
        })
        
        decile_path = prop_dir / "decile_lift.csv"
        decile_data.to_csv(decile_path, index=False)
    
    # Monkeypatch working directory
    with monkeypatch.context() as m:
        m.chdir(tmp_path)
        
        # Run comparison summary
        result_df = build_comparison_summary(k=3, algo="kmeans")
    
    # Assert comparison_summary.csv exists
    figures_dir = tmp_path / "reports" / "figures"
    summary_path = figures_dir / "comparison_summary.csv"
    assert summary_path.exists()
    
    # Load and check summary
    summary_df = pd.read_csv(summary_path)
    
    # Check expected columns exist
    expected_cols = [
        "prefix", "algo", "k", "silhouette", "calinski_harabasz", 
        "davies_bouldin", "valid_size", "propensity_auc", 
        "propensity_pr_auc", "propensity_brier", "top_decile_lift"
    ]
    
    for col in expected_cols:
        assert col in summary_df.columns
    
    # Check we have both prefixes
    assert len(summary_df) == 2
    assert set(summary_df["prefix"]) == {"with_sentiment", "no_sentiment"}
    
    # Check values are populated
    assert summary_df["silhouette"].notna().all()
    assert summary_df["propensity_auc"].notna().all()
    assert summary_df["top_decile_lift"].notna().all()
    
    # Check returned DataFrame matches saved file
    pd.testing.assert_frame_equal(result_df, summary_df)


@pytest.mark.xfail(reason="Skip if SHAP not installed")
def test_shap_optional_dependency():
    """Test that SHAP dependency is optional."""
    try:
        import shap
        pytest.skip("SHAP is installed")
    except ImportError:
        # This is expected when SHAP is not installed
        pass