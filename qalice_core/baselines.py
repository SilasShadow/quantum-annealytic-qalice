"""Clustering baselines for bank marketing feature views."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


def _load_feature_data(prefix: str) -> Tuple[pd.DataFrame, dict, dict]:
    """Load feature view, schema, and splits for given prefix."""
    base_path = Path("data/processed/bank_marketing")

    # Load feature view
    feature_path = base_path / f"{prefix}_feature_view.parquet"
    df = pd.read_parquet(feature_path)
    logger.info(f"Loaded {len(df)} rows from {feature_path}")

    # Load schema
    schema_path = base_path / f"{prefix}_feature_schema.json"
    with open(schema_path) as f:
        schema = json.load(f)

    # Load splits
    splits_path = base_path / "splits.json"
    with open(splits_path) as f:
        splits = json.load(f)

    return df, schema, splits


def _prepare_modeling_data(
    df: pd.DataFrame, schema: dict, splits: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare train/valid splits with modeling features only."""
    # Get modeling features (use scaled numeric features + one-hot features)
    modeling_features = schema["one_hot_features"] + schema["scaler_features"]
    passthrough_cols = ["y", "month", "month_idx", "cons.conf.idx"]
    modeling_features = [f for f in modeling_features if f not in passthrough_cols]

    # Split by month buckets
    train_mask = df["month_idx"].isin(splits["train_months"])
    valid_mask = df["month_idx"].isin(splits["valid_months"])

    train_df = df[train_mask][modeling_features].copy()
    valid_df = df[valid_mask][modeling_features].copy()

    logger.info(f"Train features: {len(train_df)} rows, {len(modeling_features)} features")
    logger.info(f"Valid features: {len(valid_df)} rows, {len(modeling_features)} features")

    return train_df, valid_df


def _compute_cluster_metrics(features: pd.DataFrame, labels: np.ndarray) -> dict:
    """Compute clustering metrics on validation set."""
    metrics = {}

    # Silhouette score (only if enough samples)
    if len(features) >= 100:
        try:
            metrics["silhouette"] = silhouette_score(features, labels)
        except Exception:
            metrics["silhouette"] = None
    else:
        metrics["silhouette"] = None

    # Other metrics
    try:
        metrics["calinski_harabasz"] = calinski_harabasz_score(features, labels)
        metrics["davies_bouldin"] = davies_bouldin_score(features, labels)
    except Exception:
        metrics["calinski_harabasz"] = None
        metrics["davies_bouldin"] = None

    return metrics


def _build_kpi_table(df: pd.DataFrame, labels: np.ndarray, prefix: str) -> pd.DataFrame:
    """Build KPI table for validation set with cluster statistics."""
    # Load the interim data which has both original features and month_idx
    interim_df = pd.read_csv("data/interim/bank_marketing_clean.csv")

    # Load splits to get valid indices
    base_path = Path("data/processed/bank_marketing")
    with open(base_path / "splits.json") as f:
        splits = json.load(f)

    valid_mask = interim_df["month_idx"].isin(splits["valid_months"])
    valid_full = interim_df[valid_mask].copy().reset_index(drop=True)
    valid_full["cluster_id"] = labels

    kpi_rows = []
    for cluster_id in sorted(valid_full["cluster_id"].unique()):
        cluster_data = valid_full[valid_full["cluster_id"] == cluster_id]

        # Basic stats
        size = len(cluster_data)
        size_pct = size / len(valid_full) * 100
        conversion_rate = (cluster_data["y"] == "yes").mean()  # Convert yes/no to numeric
        avg_age = cluster_data["age"].mean()
        avg_campaign = cluster_data["campaign"].mean()
        avg_pdays = cluster_data["pdays"].mean()
        avg_previous = cluster_data["previous"].mean()

        # Top 3 categoricals
        job_mode = (
            cluster_data["job"].mode().iloc[0] if len(cluster_data["job"].mode()) > 0 else "unknown"
        )
        marital_mode = (
            cluster_data["marital"].mode().iloc[0]
            if len(cluster_data["marital"].mode()) > 0
            else "unknown"
        )
        education_mode = (
            cluster_data["education"].mode().iloc[0]
            if len(cluster_data["education"].mode()) > 0
            else "unknown"
        )

        job_count = (cluster_data["job"] == job_mode).sum()
        marital_count = (cluster_data["marital"] == marital_mode).sum()
        education_count = (cluster_data["education"] == education_mode).sum()

        kpi_rows.append(
            {
                "cluster_id": cluster_id,
                "size": size,
                "size_pct": size_pct,
                "conversion_rate": conversion_rate,
                "avg_age": avg_age,
                "avg_campaign": avg_campaign,
                "avg_pdays": avg_pdays,
                "avg_previous": avg_previous,
                "top_job": f"{job_mode}({job_count})",
                "top_marital": f"{marital_mode}({marital_count})",
                "top_education": f"{education_mode}({education_count})",
            }
        )

    return pd.DataFrame(kpi_rows)


def _create_visualization(
    features: pd.DataFrame, labels: np.ndarray, algo: str, k: int, prefix: str, output_dir: Path
) -> None:
    """Create UMAP/t-SNE visualization of clusters."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping visualization")
        return

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(features)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE

        perplexity = min(30, len(features) // 3)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embedding = reducer.fit_transform(features)
        method = "t-SNE"

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, alpha=0.7, s=20, cmap="tab10")
    plt.colorbar(scatter)
    plt.title(f"{prefix} | {algo} | k={k} | {method}")
    plt.axis("off")

    output_path = output_dir / f"{algo}_k{k}_umap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved visualization: {output_path}")


def _run_single_clustering(prefix: str, k_list: List[int]) -> pd.DataFrame:
    """Run clustering for a single prefix (with_sentiment or no_sentiment)."""
    # Load data
    df, schema, splits = _load_feature_data(prefix)
    train_features, valid_features = _prepare_modeling_data(df, schema, splits)

    # Create output directory
    output_dir = Path(f"reports/baselines/{prefix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for k in k_list:
        for algo_name, algo_class in [("kmeans", KMeans), ("gmm", GaussianMixture)]:
            logger.info(f"Running {algo_name} with k={k} for {prefix}")

            # Fit model
            if algo_name == "kmeans":
                model = algo_class(n_clusters=k, random_state=42, n_init=10)
            else:  # GMM
                model = algo_class(n_components=k, random_state=42)

            model.fit(train_features)

            # Predict clusters
            train_labels = model.predict(train_features)
            valid_labels = model.predict(valid_features)

            # Compute metrics on validation set
            metrics = _compute_cluster_metrics(valid_features, valid_labels)
            metrics.update({"n_train": len(train_features), "n_valid": len(valid_features)})

            # Build KPI table
            kpi_df = _build_kpi_table(df, valid_labels, prefix)

            # Create assignments DataFrame
            train_assignments = pd.DataFrame({"cluster_id": train_labels, "split": "train"})
            valid_assignments = pd.DataFrame({"cluster_id": valid_labels, "split": "valid"})
            assignments_df = pd.concat([train_assignments, valid_assignments], ignore_index=True)

            # Save artifacts
            assignments_path = output_dir / f"{algo_name}_k{k}_assignments.parquet"
            assignments_df.to_parquet(assignments_path, index=True)

            kpi_path = output_dir / f"{algo_name}_k{k}_kpi_valid.csv"
            kpi_df.to_csv(kpi_path, index=False)

            metrics_path = output_dir / f"{algo_name}_k{k}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            # Create visualization
            _create_visualization(valid_features, valid_labels, algo_name, k, prefix, output_dir)

            logger.info(f"Saved artifacts for {algo_name}_k{k} to {output_dir}")

            # Add to summary
            summary_rows.append(
                {
                    "algo": algo_name,
                    "k": k,
                    "silhouette": metrics["silhouette"],
                    "calinski_harabasz": metrics["calinski_harabasz"],
                    "davies_bouldin": metrics["davies_bouldin"],
                    "valid_size": metrics["n_valid"],
                }
            )

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary: {summary_path}")

    return summary_df


def run_clustering(prefix: str, k_list: Optional[List[int]] = None) -> pd.DataFrame:
    """Run clustering baselines for a specific prefix.

    Args:
        prefix: Either "with_sentiment" or "no_sentiment"
        k_list: List of k values to test. Defaults to [8]

    Returns:
        DataFrame with metrics for all algorithm/k combinations
    """
    if k_list is None:
        k_list = [8]

    if prefix not in ["with_sentiment", "no_sentiment"]:
        raise ValueError(f"Invalid prefix: {prefix}. Must be 'with_sentiment' or 'no_sentiment'")

    return _run_single_clustering(prefix, k_list)


def run_all(k_list: Optional[List[int]] = None) -> pd.DataFrame:
    """Run clustering baselines for both sentiment prefixes.

    Args:
        k_list: List of k values to test. Defaults to [8]

    Returns:
        DataFrame with metrics for all prefix/algorithm/k combinations
    """
    if k_list is None:
        k_list = [8]

    all_results = []

    for prefix in ["with_sentiment", "no_sentiment"]:
        logger.info(f"Processing {prefix} features")
        result_df = run_clustering(prefix, k_list)
        result_df["prefix"] = prefix
        all_results.append(result_df)

    combined_df = pd.concat(all_results, ignore_index=True)
    return combined_df
