"""Feature toggle constants and file naming utilities."""

SENTIMENT_FEATURE = "cons.conf.idx"


def prefix(include_sentiment: bool) -> str:
    """Get file prefix based on sentiment inclusion.

    Args:
        include_sentiment: Whether sentiment feature is included

    Returns:
        File prefix string
    """
    return "with_sentiment" if include_sentiment else "no_sentiment"


def fv_path(include_sentiment: bool) -> str:
    """Get feature view parquet file path.

    Args:
        include_sentiment: Whether sentiment feature is included

    Returns:
        Path to feature view parquet file
    """
    return f"data/processed/bank_marketing/{prefix(include_sentiment)}_feature_view.parquet"


def schema_path(include_sentiment: bool) -> str:
    """Get feature schema JSON file path.

    Args:
        include_sentiment: Whether sentiment feature is included

    Returns:
        Path to feature schema JSON file
    """
    return f"data/processed/bank_marketing/{prefix(include_sentiment)}_feature_schema.json"


def scaler_path(include_sentiment: bool) -> str:
    """Get scaler pickle file path.

    Args:
        include_sentiment: Whether sentiment feature is included

    Returns:
        Path to scaler pickle file
    """
    return f"data/processed/bank_marketing/scaler_{prefix(include_sentiment)}.pkl"
