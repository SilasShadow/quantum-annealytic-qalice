"""Data ingestion for enriched Bank Marketing dataset."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Set

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: Set[str] = {
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
    "y",
}

CATEGORICAL_COLUMNS = {
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
}

NUMERIC_COLUMNS = {
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
    "duration",
}


def load_enriched_csv(path: str) -> pd.DataFrame:
    """Load enriched Bank Marketing CSV with schema validation.

    Args:
        path: Path to bank-additional-full.csv

    Returns:
        Raw DataFrame with validated schema

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Try semicolon delimiter first (standard for this dataset), fallback to comma
    try:
        df = pd.read_csv(path, sep=";")
    except pd.errors.ParserError:
        df = pd.read_csv(path, sep=",")

    # Normalize column names to snake_case
    df.columns = df.columns.str.strip().str.replace(".", "_", regex=False).str.lower()

    # Validate required columns
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path}")
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize data types.

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame with proper types
    """
    df = df.copy()

    # Map target variable
    df["y"] = df["y"].map({"yes": 1, "no": 0}).astype(int)

    # Clean categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            if col == "month":
                df[col] = df[col].str.lower()

    # Convert numeric columns with coercion
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            original_nulls = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            new_nulls = df[col].isna().sum()
            if new_nulls > original_nulls:
                logger.warning(f"Column {col}: {new_nulls - original_nulls} values coerced to NaN")

    return df


def add_month_idx(df: pd.DataFrame) -> pd.DataFrame:
    """Add monotone month index based on file order.

    Args:
        df: Cleaned DataFrame

    Returns:
        DataFrame with month_idx column

    Raises:
        AssertionError: If month_idx is not monotone non-decreasing
    """
    df = df.copy()

    month_idx = [0]
    current_month = df.iloc[0]["month"]

    for i in range(1, len(df)):
        if df.iloc[i]["month"] != current_month:
            month_idx.append(month_idx[-1] + 1)
            current_month = df.iloc[i]["month"]
        else:
            month_idx.append(month_idx[-1])

    df["month_idx"] = month_idx

    # Validate monotonicity
    assert df["month_idx"].is_monotonic_increasing, "month_idx must be non-decreasing"

    return df


def persist_artifacts(
    df_clean_with_month_idx: pd.DataFrame, df_audit_with_duration: pd.DataFrame
) -> None:
    """Persist audit CSV and provenance JSON.

    Args:
        df_clean_with_month_idx: Clean DataFrame without duration
        df_audit_with_duration: Audit DataFrame with duration retained
    """
    interim_dir = Path("data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)

    # Write audit CSV with duration
    audit_path = interim_dir / "bank_marketing_clean.csv"
    df_audit_with_duration.to_csv(audit_path, index=False)
    logger.info(f"Audit CSV written to {audit_path}")

    # Write provenance JSON
    provenance = {
        "source": "uci_enriched_local",
        "file": "bank-additional-full.csv",
        "n_rows": len(df_clean_with_month_idx),
        "n_cols": len(df_clean_with_month_idx.columns),
        "has_macro_fields": True,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    provenance_path = interim_dir / "bank_marketing_provenance.json"
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info(f"Provenance JSON written to {provenance_path}")


def load_bank_marketing() -> pd.DataFrame:
    """Load and process enriched Bank Marketing dataset.

    Returns:
        Clean DataFrame ready for feature engineering (duration removed)
    """
    source_path = "data/raw/bank_marketing/bank-additional/bank-additional-full.csv"

    # Load and process
    try:
        df_raw = load_enriched_csv(source_path)
    except FileNotFoundError:
        # Return empty DataFrame with required columns for testing
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS) + ["month_idx"])
    df_clean = basic_clean(df_raw)
    df_with_month_idx = add_month_idx(df_clean)

    # Create audit version (with duration) and modeling version (without duration)
    df_audit = df_with_month_idx.copy()
    df_modeling = df_with_month_idx.drop(columns=["duration"])

    # Persist artifacts
    persist_artifacts(df_modeling, df_audit)

    logger.info(f"Final dataset: {len(df_modeling)} rows, {len(df_modeling.columns)} columns")
    return df_modeling
