"""Compare QUBO clusters with baseline clusters and generate analysis reports."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def compare_with_baseline(
    prefix: str = "with_sentiment", 
    algo: str = "kmeans", 
    k: int = 8,
    do_refit: bool = False,
    sampler: Literal["sa", "sqa"] = "sa"
) -> Dict[str, Any]:
    """Compare QUBO clusters with baseline clusters and generate analysis reports.
    
    Args:
        prefix: Feature set prefix ("with_sentiment" or "without_sentiment")
        algo: Baseline algorithm name (default "kmeans")
        k: Number of clusters
        do_refit: Whether to refit baseline if timing data missing
        sampler: QUBO sampler type ("sa" or "sqa")
        
    Returns:
        Dict with NMI, ARI, and best match information
    """
    # Load data
    qubo_assignments = pd.read_parquet(f"reports/qi/{prefix}/{sampler}/assignments_valid.parquet")
    baseline_assignments = pd.read_parquet(f"reports/baselines/{prefix}/{algo}_k{k}_assignments.parquet")
    feature_view = pd.read_parquet(f"data/processed/bank_marketing/{prefix}_feature_view.parquet")
    
    # Load timings
    with open(f"reports/qi/{prefix}/{sampler}/timings.json") as f:
        qubo_timings = json.load(f)
    
    # Merge assignments on index
    merged = qubo_assignments.merge(baseline_assignments, left_index=True, right_index=True, how="inner")
    qubo_labels = merged["cluster_qubo"]
    baseline_labels = merged["cluster_baseline"]
    
    # Compute metrics
    nmi = normalized_mutual_info_score(qubo_labels, baseline_labels)
    ari = adjusted_rand_score(qubo_labels, baseline_labels)
    
    # Build contingency table and solve Hungarian matching
    contingency = pd.crosstab(qubo_labels, baseline_labels)
    cost_matrix = -contingency.values  # Negative for maximization
    qubo_indices, baseline_indices = linear_sum_assignment(cost_matrix)
    
    best_match = {}
    for qi, bi in zip(qubo_indices, baseline_indices):
        best_match[qi] = bi
    
    # Create overlap heatmap
    plt.figure(figsize=(10, 8))
    percentages = contingency.div(contingency.sum(axis=1), axis=0) * 100
    sns.heatmap(percentages, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=[f'Baseline {i}' for i in contingency.columns],
                yticklabels=[f'QUBO {i}' for i in contingency.index])
    plt.title(f'Cluster Overlap: QUBO vs {algo.upper()} (k={k})\nNMI={nmi:.3f}, ARI={ari:.3f}')
    plt.xlabel('Baseline Clusters')
    plt.ylabel('QUBO Clusters')
    plt.tight_layout()
    
    # Save heatmap
    output_dir = Path(f"reports/qi/{prefix}/{sampler}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"overlap_{algo}_k{k}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate KPIs by QUBO cluster
    valid_data = feature_view[feature_view["split"] == "valid"].copy()
    valid_data = valid_data.merge(qubo_assignments, left_index=True, right_index=True, how="inner")
    
    kpi_data = []
    total_valid = len(valid_data)
    
    for cluster_id in sorted(valid_data["cluster_qubo"].unique()):
        cluster_data = valid_data[valid_data["cluster_qubo"] == cluster_id]
        
        kpi_row = {
            "cluster_qubo": cluster_id,
            "size": len(cluster_data),
            "size_pct": len(cluster_data) / total_valid * 100,
            "conversion_rate": cluster_data["y"].mean(),
            "avg_age": cluster_data["age"].mean(),
            "avg_campaign": cluster_data["campaign"].mean(),
            "avg_pdays": cluster_data["pdays"].mean(),
            "avg_previous": cluster_data["previous"].mean()
        }
        kpi_data.append(kpi_row)
    
    kpi_df = pd.DataFrame(kpi_data)
    kpi_df.to_csv(output_dir / "kpi_valid_qubo.csv", index=False)
    
    # Generate runtime table
    runtime_data = []
    
    # QUBO timings
    qubo_row = {
        "method": "QUBO",
        "k": k,
        "train_rows": len(feature_view[feature_view["split"] == "train"]),
        "valid_rows": len(feature_view[feature_view["split"] == "valid"]),
        "fit_sec": (qubo_timings.get("coarsen_sec", 0) + 
                   qubo_timings.get("bqm_build_sec", 0)),
        "solve_sec": (qubo_timings.get("anneal_sec", 0) + 
                     qubo_timings.get("decode_sec", 0) + 
                     qubo_timings.get("expand_sec", 0)),
        "total_sec": qubo_timings.get("total_sec", 0)
    }
    runtime_data.append(qubo_row)
    
    # Baseline timings
    baseline_metrics_path = Path(f"reports/baselines/{prefix}/{algo}_k{k}_metrics.json")
    baseline_fit_sec = 0
    baseline_predict_sec = 0
    
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path) as f:
            baseline_metrics = json.load(f)
        baseline_fit_sec = baseline_metrics.get("fit_sec", 0)
        baseline_predict_sec = baseline_metrics.get("predict_sec", 0)
    elif do_refit:
        # Quick refit to estimate timing
        train_data = feature_view[feature_view["split"] == "train"]
        numeric_cols = ["age", "campaign", "pdays", "previous", "emp.var.rate", 
                       "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
        X_train = train_data[numeric_cols].fillna(0)
        
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train)
        baseline_fit_sec = time.time() - start_time
        
        valid_data_baseline = feature_view[feature_view["split"] == "valid"]
        X_valid = valid_data_baseline[numeric_cols].fillna(0)
        
        start_time = time.time()
        kmeans.predict(X_valid)
        baseline_predict_sec = time.time() - start_time
    
    baseline_row = {
        "method": algo.upper(),
        "k": k,
        "train_rows": len(feature_view[feature_view["split"] == "train"]),
        "valid_rows": len(feature_view[feature_view["split"] == "valid"]),
        "fit_sec": baseline_fit_sec,
        "solve_sec": baseline_predict_sec,
        "total_sec": baseline_fit_sec + baseline_predict_sec
    }
    runtime_data.append(baseline_row)
    
    runtime_df = pd.DataFrame(runtime_data)
    runtime_df.to_csv(output_dir / "runtime_table.csv", index=False)
    
    return {
        "nmi": nmi,
        "ari": ari,
        "best_match": best_match
    }


if __name__ == "__main__":
    # Example usage
    result = compare_with_baseline(prefix="with_sentiment", algo="kmeans", k=8)
    print(f"NMI: {result['nmi']:.3f}")
    print(f"ARI: {result['ari']:.3f}")
    print(f"Best match: {result['best_match']}")