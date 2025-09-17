"""Feature engineering for Bank Marketing dataset with sentiment toggle."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

CATEGORICAL = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "poutcome",
    "month",
    "day_of_week",
]

NUMERIC_BASE = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_idx",
    "euribor3m",
    "nr_employed",
]

SENTIMENT_FEATURE = "cons_conf_idx"
PASSTHROUGH = ["y", "month", "month_idx", "cons_conf_idx"]


def build_feature_view(
    df: pd.DataFrame, include_sentiment: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any], StandardScaler]:
    """Build feature view with optional sentiment feature.

    Args:
        df: Cleaned DataFrame from data_ingest
        include_sentiment: Whether to include cons_conf_idx in modeling features

    Returns:
        Tuple of (feature_view_df, schema_dict, fitted_scaler)

    Raises:
        ValueError: If required columns are missing
    """
    # Validate required columns
    required_cols = set(CATEGORICAL + NUMERIC_BASE + [SENTIMENT_FEATURE] + PASSTHROUGH)
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    df_work = df.copy()

    # Define numeric features based on sentiment toggle
    numeric_features = NUMERIC_BASE + ([SENTIMENT_FEATURE] if include_sentiment else [])

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df_work[CATEGORICAL], prefix_sep="_", dtype=int)
    one_hot_features = df_encoded.columns.tolist()

    # Scale numeric features
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_work[numeric_features]),
        columns=[f"{col}_scaled" for col in numeric_features],
        index=df_work.index,
    )

    # Combine features and passthrough
    feature_view = pd.concat([df_encoded, df_scaled, df_work[PASSTHROUGH]], axis=1)

    # Build schema
    schema = {
        "numeric_features": numeric_features,
        "categorical_features": CATEGORICAL,
        "one_hot_features": one_hot_features,
        "sentiment_included": include_sentiment,
        "scaler_features": [f"{col}_scaled" for col in numeric_features],
        "passthrough_features": PASSTHROUGH,
    }

    logger.info(
        f"Built feature view: {len(feature_view)} rows, {len(feature_view.columns)} columns"
    )
    logger.info(
        f"One-hot features: {len(one_hot_features)}, Numeric features: {len(numeric_features)}"
    )

    return feature_view, schema, scaler


def persist_feature_view(
    fv: pd.DataFrame,
    schema: Dict[str, Any],
    include_sentiment: bool,
    fitted_scaler: StandardScaler = None,
) -> None:
    """Persist feature view and schema artifacts.

    Args:
        fv: Feature view DataFrame
        schema: Feature schema dictionary
        include_sentiment: Whether sentiment was included
        fitted_scaler: Pre-fitted scaler to save
    """
    # Create output directory
    output_dir = Path("data/processed/bank_marketing")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set prefix based on sentiment inclusion
    prefix = "with_sentiment" if include_sentiment else "no_sentiment"

    # Save feature view as parquet
    parquet_path = output_dir / f"{prefix}_feature_view.parquet"
    fv.to_parquet(parquet_path, index=False)
    logger.info(f"Feature view saved to {parquet_path}")

    # Save schema as JSON
    schema_path = output_dir / f"{prefix}_feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info(f"Feature schema saved to {schema_path}")

    # Save scaler if provided
    if fitted_scaler is not None:
        scaler_path = output_dir / f"scaler_{prefix}.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(fitted_scaler, f)
        logger.info(f"Scaler saved to {scaler_path}")


def create_both_feature_views(df: pd.DataFrame) -> None:
    """Create both sentiment and no-sentiment feature views.

    Args:
        df: Cleaned DataFrame from data_ingest
    """
    # Create with sentiment
    fv_with, schema_with = build_feature_view(df, include_sentiment=True)
    persist_feature_view(fv_with, schema_with, include_sentiment=True)

    # Create without sentiment
    fv_without, schema_without = build_feature_view(df, include_sentiment=False)
    persist_feature_view(fv_without, schema_without, include_sentiment=False)

    logger.info("Both feature views created successfully")


def _save_fitted_scaler(
    df: pd.DataFrame, numeric_features: List[str], include_sentiment: bool
) -> None:
    """Save properly fitted scaler for the given features.

    Args:
        df: Original DataFrame with raw numeric features
        numeric_features: List of numeric feature names
        include_sentiment: Whether sentiment was included
    """
    output_dir = Path("data/processed/bank_marketing")
    prefix = "with_sentiment" if include_sentiment else "no_sentiment"

    scaler = StandardScaler()
    scaler.fit(df[numeric_features])

    scaler_path = output_dir / f"scaler_{prefix}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Fitted scaler saved to {scaler_path}")


def build_and_persist_feature_view(
    df: pd.DataFrame, include_sentiment: bool = True
) -> pd.DataFrame:
    """Build and persist feature view in one call.

    Args:
        df: Cleaned DataFrame from data_ingest
        include_sentiment: Whether to include sentiment feature

    Returns:
        Feature view DataFrame
    """
    # Build feature view
    feature_view, schema, scaler = build_feature_view(df, include_sentiment)

    # Persist artifacts
    persist_feature_view(feature_view, schema, include_sentiment, scaler)

    return feature_view
