"""
Hyperparameter scanning orchestrator for QUBO balanced k-means.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Literal

import pandas as pd
from sklearn.metrics import silhouette_score

from .qubo_balanced_kmeans import run_qubo

logger = logging.getLogger(__name__)


def compute_silhouette_valid(prefix: str, X_valid: pd.DataFrame, valid_assignments: List[int]) -> float:
    """Compute silhouette score on validation set."""
    if len(set(valid_assignments)) < 2:
        return -1.0  # Invalid clustering
    
    try:
        return silhouette_score(X_valid.values, valid_assignments)
    except Exception as e:
        logger.warning(f"Failed to compute silhouette score: {e}")
        return -1.0


def compute_balance_deviation(assignments, k: int) -> float:
    """Compute balance deviation percentage."""
    import numpy as np
    assignments = np.array(assignments)
    cluster_sizes = [(assignments == i).sum() for i in range(k)]
    target_size = len(assignments) / k
    
    if target_size == 0:
        return 100.0
    
    deviations = [abs(size - target_size) / target_size for size in cluster_sizes]
    return max(deviations) * 100.0


def scan_and_select(
    prefix: str = "with_sentiment",
    k: int = 8,
    lambda1_grid: List[float] = [2.0, 5.0, 10.0],
    lambda2_grid: List[float] = [0.5, 1.0, 2.0],
    random_state: int = 42,
    sampler: Literal["sa", "sqa"] = "sa"
) -> Dict[str, Any]:
    """
    Scan hyperparameter grid and select best configuration.
    
    Returns:
        Dict with best parameters and full results
    """
    logger.info(f"Starting hyperparameter scan for prefix='{prefix}', k={k}, sampler={sampler}")
    
    # Setup output paths
    reports_path = Path("reports/qi") / prefix
    reports_path.mkdir(parents=True, exist_ok=True)
    
    # Load validation data once
    base_path = Path("data/processed/bank_marketing")
    feature_view_path = base_path / f"{prefix}_feature_view.parquet"
    schema_path = base_path / f"{prefix}_feature_schema.json"
    splits_path = base_path / "splits.json"
    
    if not all(p.exists() for p in [feature_view_path, schema_path, splits_path]):
        raise FileNotFoundError("Required input artifacts missing")
    
    df = pd.read_parquet(feature_view_path)
    with open(schema_path) as f:
        schema = json.load(f)
    with open(splits_path) as f:
        splits = json.load(f)
    
    modeling_features = schema["one_hot_features"] + schema["scaler_features"]
    X = df[modeling_features]
    valid_mask = df["month_idx"].isin(splits["valid_months"])
    X_valid = X[valid_mask]
    
    # Grid search
    results = []
    
    for lambda1 in lambda1_grid:
        for lambda2 in lambda2_grid:
            logger.info(f"Testing lambda1={lambda1}, lambda2={lambda2}")
            
            start_time = time.time()
            
            try:
                # Run QUBO with fixed parameters
                result = run_qubo(
                    prefix=prefix,
                    k=k,
                    M=400,
                    knn=40,
                    lambda1=lambda1,
                    lambda2=lambda2,
                    sweeps=10_000,
                    num_reads=64,
                    random_state=random_state,
                    sampler=sampler
                )
                
                # Extract timings
                build_sec = result["timings"]["bqm_build_sec"]
                solve_sec = result["timings"]["solve_sec"]
                total_sec = result["timings"]["total_sec"]
                
                # Compute validation silhouette
                valid_assignments = result["valid_assignments"]
                silhouette_valid = compute_silhouette_valid(prefix, X_valid, valid_assignments)
                
                # Compute balance deviation
                balance_dev_pct = compute_balance_deviation(valid_assignments, k)
                
                results.append({
                    "lambda1": lambda1,
                    "lambda2": lambda2,
                    "silhouette_valid": silhouette_valid,
                    "balance_dev_pct": balance_dev_pct,
                    "build_sec": build_sec,
                    "solve_sec": solve_sec,
                    "total_sec": total_sec,
                    "sampler": sampler,
                    "selected": False
                })
                
                logger.info(f"  → silhouette={silhouette_valid:.4f}, balance_dev={balance_dev_pct:.2f}%")
                
            except Exception as e:
                logger.error(f"Failed for lambda1={lambda1}, lambda2={lambda2}: {e}")
                results.append({
                    "lambda1": lambda1,
                    "lambda2": lambda2,
                    "silhouette_valid": -1.0,
                    "balance_dev_pct": 100.0,
                    "build_sec": 0.0,
                    "solve_sec": 0.0,
                    "total_sec": time.time() - start_time,
                    "sampler": sampler,
                    "selected": False
                })
    
    # Select best configuration
    # 1) Feasible (balance_dev_pct <= 5%)
    # 2) Highest silhouette_valid
    # 3) Lowest total_sec
    
    feasible_results = [r for r in results if r["balance_dev_pct"] <= 5.0]
    
    if feasible_results:
        # Sort by silhouette (desc), then total_sec (asc)
        best_result = max(feasible_results, key=lambda x: (x["silhouette_valid"], -x["total_sec"]))
    else:
        logger.warning("No feasible solutions found, selecting best by silhouette score")
        best_result = max(results, key=lambda x: x["silhouette_valid"])
    
    # Mark selected
    for r in results:
        if (r["lambda1"] == best_result["lambda1"] and 
            r["lambda2"] == best_result["lambda2"]):
            r["selected"] = True
            break
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(reports_path / "model_selection.csv", index=False)
    
    best_params = {
        "lambda1": best_result["lambda1"],
        "lambda2": best_result["lambda2"],
        "k": k,
        "sampler": sampler,
        "selection_criteria": {
            "feasible_count": len(feasible_results),
            "total_count": len(results),
            "best_silhouette": best_result["silhouette_valid"],
            "best_balance_dev": best_result["balance_dev_pct"]
        }
    }
    
    with open(reports_path / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    logger.info(f"Scan complete. Best: lambda1={best_result['lambda1']}, lambda2={best_result['lambda2']}")
    logger.info(f"Results saved to {reports_path}")
    
    return {
        "best_params": best_params,
        "all_results": results,
        "selected_result": best_result
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run for both prefixes and samplers
    for prefix in ["with_sentiment", "no_sentiment"]:
        for sampler in ["sa", "sqa"]:
            try:
                result = scan_and_select(prefix, sampler=sampler)
                print(f"✓ Completed hyperparameter scan for {prefix} with {sampler}")
                print(f"  Best: λ1={result['best_params']['lambda1']}, λ2={result['best_params']['lambda2']}")
            except Exception as e:
                print(f"✗ Failed for {prefix} with {sampler}: {e}")