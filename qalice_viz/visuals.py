"""Business-facing visualizations and comparison tables."""

import json
import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def build_kpi_bars(prefix: str, algo: str = "kmeans", k: int = 8) -> Tuple[str, str]:
    """Build conversion rate and cluster size bar charts.

    Args:
        prefix: Feature prefix (with_sentiment or no_sentiment)
        algo: Algorithm name (default: kmeans)
        k: Number of clusters (default: 8)

    Returns:
        Tuple of (conversion_rate_path, size_bars_path)
    """
    # Read KPI data
    kpi_path = Path(f"reports/baselines/{prefix}/{algo}_k{k}_kpi_valid.csv")
    if not kpi_path.exists():
        raise FileNotFoundError(f"KPI file not found: {kpi_path}")

    kpi_df = pd.read_csv(kpi_path)

    # Create output directory
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Conversion rate bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(kpi_df["cluster_id"], kpi_df["conversion_rate"])
    plt.xlabel("Cluster ID")
    plt.ylabel("Conversion Rate")
    plt.title(f"Conversion Rate by Cluster ({prefix})")
    plt.xticks(kpi_df["cluster_id"])
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    cr_path = output_dir / f"{prefix}_{algo}_k{k}_cr_bars.png"
    plt.savefig(cr_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved conversion rate bars: {cr_path}")

    # Cluster size bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(kpi_df["cluster_id"], kpi_df["size_pct"])
    plt.xlabel("Cluster ID")
    plt.ylabel("Cluster Size (%)")
    plt.title(f"Cluster Size Distribution ({prefix})")
    plt.xticks(kpi_df["cluster_id"])
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    size_path = output_dir / f"{prefix}_{algo}_k{k}_size_bars.png"
    plt.savefig(size_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved cluster size bars: {size_path}")

    return str(cr_path), str(size_path)


def build_comparison_summary(k: int = 8, algo: str = "kmeans") -> pd.DataFrame:
    """Build comparison summary across prefixes.

    Args:
        k: Number of clusters (default: 8)
        algo: Algorithm name (default: kmeans)

    Returns:
        DataFrame with comparison metrics
    """
    prefixes = ["with_sentiment", "no_sentiment"]
    rows = []

    for prefix in prefixes:
        row = {"prefix": prefix, "algo": algo, "k": k}

        # Load clustering metrics
        metrics_path = Path(f"reports/baselines/{prefix}/{algo}_k{k}_metrics.json")
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            row.update(
                {
                    "silhouette": metrics.get("silhouette"),
                    "calinski_harabasz": metrics.get("calinski_harabasz"),
                    "davies_bouldin": metrics.get("davies_bouldin"),
                    "valid_size": metrics.get("n_valid"),
                }
            )
        else:
            row.update(
                {
                    "silhouette": None,
                    "calinski_harabasz": None,
                    "davies_bouldin": None,
                    "valid_size": None,
                }
            )

        # Load propensity metrics
        prop_metrics_path = Path(f"reports/propensity/{prefix}/metrics.json")
        if prop_metrics_path.exists():
            with open(prop_metrics_path) as f:
                prop_metrics = json.load(f)
            row.update(
                {
                    "propensity_auc": prop_metrics.get("auc"),
                    "propensity_pr_auc": prop_metrics.get("pr_auc"),
                    "propensity_brier": prop_metrics.get("brier"),
                }
            )
        else:
            row.update(
                {
                    "propensity_auc": None,
                    "propensity_pr_auc": None,
                    "propensity_brier": None,
                }
            )

        # Load top decile lift
        decile_path = Path(f"reports/propensity/{prefix}/decile_lift.csv")
        if decile_path.exists():
            decile_df = pd.read_csv(decile_path)
            top_decile_lift = decile_df[decile_df["decile"] == 1][
                "lift_vs_overall"
            ].iloc[0]
            row["top_decile_lift"] = top_decile_lift
        else:
            row["top_decile_lift"] = None

        rows.append(row)

    comparison_df = pd.DataFrame(rows)

    # Save comparison table
    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comparison_summary.csv"
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"Saved comparison summary: {output_path}")

    return comparison_df
