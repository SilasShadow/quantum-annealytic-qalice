"""Tests for CLI Stage-1 command."""

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from qalice_core.cli import app

pytestmark = pytest.mark.skip(reason="CLI tests disabled temporarily")


@pytest.fixture
def mock_df():
    """Create mock DataFrame for testing."""
    return pd.DataFrame(
        {
            # Client data
            "age": [25, 30, 35, 40],
            "job": ["admin", "technician", "management", "admin"],
            "marital": ["single", "married", "single", "married"],
            "education": ["university.degree", "high.school", "university.degree", "high.school"],
            "default": ["no", "no", "no", "no"],
            "housing": ["yes", "no", "yes", "no"],
            "loan": ["no", "no", "yes", "no"],
            # Contact data
            "contact": ["cellular", "telephone", "cellular", "telephone"],
            "month": ["jan", "jan", "feb", "feb"],
            "day_of_week": ["mon", "tue", "wed", "thu"],
            # Campaign data
            "campaign": [1, 2, 1, 3],
            "pdays": [999, 999, 5, 999],
            "previous": [0, 0, 1, 0],
            "poutcome": ["nonexistent", "nonexistent", "success", "nonexistent"],
            # Macro data
            "emp_var_rate": [1.1, 1.1, 1.4, 1.4],
            "cons_price_idx": [93.994, 93.994, 93.918, 93.918],
            "cons_conf_idx": [-36.4, -36.4, -42.7, -42.7],
            "euribor3m": [4.857, 4.857, 4.961, 4.961],
            "nr_employed": [5191.0, 5191.0, 5228.1, 5228.1],
            # Target and time
            "y": [0, 1, 0, 1],
            "month_idx": [0, 0, 1, 1],
        }
    )


def test_cli_stage1_on_mode(mock_df, tmp_path, monkeypatch):
    """Test CLI stage1 command in 'on' mode."""
    # Change to tmp directory
    monkeypatch.chdir(tmp_path)

    # Mock load_bank_marketing to return our test data
    def mock_load_bank_marketing():
        return mock_df

    monkeypatch.setattr("qalice_core.cli.load_bank_marketing", mock_load_bank_marketing)

    # Run CLI command
    runner = CliRunner()
    result = runner.invoke(app, ["stage1", "--mode", "on"])

    assert result.exit_code == 0

    # Check that with_sentiment files are created
    base_path = Path("data/processed/bank_marketing")
    assert (base_path / "with_sentiment_feature_view.parquet").exists()
    assert (base_path / "with_sentiment_feature_schema.json").exists()
    assert (base_path / "scaler_with_sentiment.pkl").exists()

    # Check that no_sentiment files are NOT created
    assert not (base_path / "no_sentiment_feature_view.parquet").exists()
    assert not (base_path / "no_sentiment_feature_schema.json").exists()
    assert not (base_path / "scaler_no_sentiment.pkl").exists()

    # Check splits.json is created
    assert (base_path / "splits.json").exists()


def test_cli_stage1_both_mode(mock_df, tmp_path, monkeypatch):
    """Test CLI stage1 command in 'both' mode."""
    monkeypatch.chdir(tmp_path)

    def mock_load_bank_marketing():
        return mock_df

    monkeypatch.setattr("qalice_core.cli.load_bank_marketing", mock_load_bank_marketing)

    runner = CliRunner()
    result = runner.invoke(app, ["stage1", "--mode", "both"])

    assert result.exit_code == 0

    # Check that both sentiment modes are created
    base_path = Path("data/processed/bank_marketing")

    # With sentiment files
    assert (base_path / "with_sentiment_feature_view.parquet").exists()
    assert (base_path / "with_sentiment_feature_schema.json").exists()
    assert (base_path / "scaler_with_sentiment.pkl").exists()

    # Without sentiment files
    assert (base_path / "no_sentiment_feature_view.parquet").exists()
    assert (base_path / "no_sentiment_feature_schema.json").exists()
    assert (base_path / "scaler_no_sentiment.pkl").exists()

    # Splits file
    assert (base_path / "splits.json").exists()


def test_cli_stage1_off_mode(mock_df, tmp_path, monkeypatch):
    """Test CLI stage1 command in 'off' mode."""
    monkeypatch.chdir(tmp_path)

    def mock_load_bank_marketing():
        return mock_df

    monkeypatch.setattr("qalice_core.cli.load_bank_marketing", mock_load_bank_marketing)

    runner = CliRunner()
    result = runner.invoke(app, ["stage1", "--mode", "off"])

    assert result.exit_code == 0

    # Check that no_sentiment files are created
    base_path = Path("data/processed/bank_marketing")
    assert (base_path / "no_sentiment_feature_view.parquet").exists()
    assert (base_path / "no_sentiment_feature_schema.json").exists()
    assert (base_path / "scaler_no_sentiment.pkl").exists()

    # Check that with_sentiment files are NOT created
    assert not (base_path / "with_sentiment_feature_view.parquet").exists()
    assert not (base_path / "with_sentiment_feature_schema.json").exists()
    assert not (base_path / "scaler_with_sentiment.pkl").exists()

    # Check splits.json is created
    assert (base_path / "splits.json").exists()


def test_schema_content(mock_df, tmp_path, monkeypatch):
    """Test that schema files contain correct sentiment information."""
    monkeypatch.chdir(tmp_path)

    def mock_load_bank_marketing():
        return mock_df

    monkeypatch.setattr("qalice_core.cli.load_bank_marketing", mock_load_bank_marketing)

    runner = CliRunner()
    result = runner.invoke(app, ["stage1", "--mode", "both"])

    assert result.exit_code == 0

    # Check with_sentiment schema
    with open("data/processed/bank_marketing/with_sentiment_feature_schema.json") as f:
        with_schema = json.load(f)
    assert with_schema["sentiment_included"] is True
    assert "cons_conf_idx" in with_schema["numeric_features"]

    # Check no_sentiment schema
    with open("data/processed/bank_marketing/no_sentiment_feature_schema.json") as f:
        no_schema = json.load(f)
    assert no_schema["sentiment_included"] is False
    assert "cons_conf_idx" not in no_schema["numeric_features"]
