"""Runtime and sentiment analysis visualization utilities."""

import json
import os
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd


def build_runtime_bar(
    prefix: str = "with_sentiment", sampler: Literal["sa", "sqa"] = "sa"
) -> str:
    """Generate runtime bar chart comparing KMeans vs QUBO stages.

    Args:
        prefix: Data prefix (with_sentiment/no_sentiment)
        sampler: QUBO sampler type ("sa" or "sqa")

    Returns:
        Path to generated PNG file
    """
    # Load QUBO timings
    qi_path = f"reports/qi/{prefix}/{sampler}/timings.json"
    if not os.path.exists(qi_path):
        raise FileNotFoundError(f"QUBO timings not found: {qi_path}")

    with open(qi_path) as f:
        timings = json.load(f)

    # Load runtime table if available, otherwise use baseline estimate
    runtime_path = f"reports/qi/{prefix}/{sampler}/runtime_table.csv"
    if os.path.exists(runtime_path):
        runtime_df = pd.read_csv(runtime_path)
        kmeans_row = runtime_df[runtime_df["method"] == "KMEANS"]
        kmeans_time = kmeans_row["total_sec"].iloc[0] if not kmeans_row.empty else 1.0
    else:
        kmeans_time = 1.0

    # Prepare data
    methods = [
        "KMeans",
        "QUBO:Coarsen",
        "QUBO:Build",
        "QUBO:Anneal",
        "QUBO:Decode",
        "QUBO:Expand",
        "QUBO:Total",
    ]
    times = [
        kmeans_time,
        timings["coarsen_sec"],
        timings["bqm_build_sec"],
        timings["anneal_sec"],
        timings["decode_sec"],
        timings["expand_sec"],
        timings["total_sec"],
    ]

    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        methods,
        times,
        color=[
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
        ],
    )

    plt.title(
        f'Runtime Comparison: {prefix.replace("_", " ").title()} ({sampler.upper()})'
    )
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45, ha="right")

    # Add value labels on bars
    for bar, time in zip(bars, times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{time:.2f}s",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # Save figure
    output_path = f"reports/figures/runtime_bar_{prefix}_{sampler}.png"
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


def build_sentiment_toggle_summary(algo: str = "kmeans", k: int = 8) -> pd.DataFrame:
    """Build sentiment toggle comparison table.

    Args:
        algo: Algorithm name (kmeans/gmm)
        k: Number of clusters

    Returns:
        DataFrame with sentiment comparison metrics
    """
    results = []

    for prefix in ["with_sentiment", "no_sentiment"]:
        # Load baseline metrics
        baseline_path = f"reports/baselines/{prefix}/summary.csv"
        baseline_df = pd.read_csv(baseline_path)
        algo_row = baseline_df[baseline_df["algo"] == algo].iloc[0]

        # Load propensity metrics
        propensity_path = f"reports/propensity/{prefix}/metrics.json"
        with open(propensity_path) as f:
            prop_metrics = json.load(f)

        # Load decile lift for top decile
        decile_path = f"reports/propensity/{prefix}/decile_lift.csv"
        decile_df = pd.read_csv(decile_path)
        top_decile_lift = decile_df.iloc[0]["lift_vs_overall"]

        results.append(
            {
                "prefix": prefix,
                "algo": algo,
                "silhouette": algo_row["silhouette"],
                "propensity_auc": prop_metrics["auc"],
                "top_decile_lift": top_decile_lift,
            }
        )

    # Check for QUBO results
    for prefix in ["with_sentiment", "no_sentiment"]:
        qi_path = f"reports/qi/{prefix}/timings.json"
        if os.path.exists(qi_path):
            results.append(
                {
                    "prefix": prefix,
                    "algo": "qubo",
                    "silhouette": "N/A",
                    "propensity_auc": "N/A",
                    "top_decile_lift": "N/A",
                }
            )

    df = pd.DataFrame(results)

    # Save to CSV
    output_path = "reports/figures/sentiment_toggle_summary.csv"
    os.makedirs("reports/figures", exist_ok=True)
    df.to_csv(output_path, index=False)

    return df
