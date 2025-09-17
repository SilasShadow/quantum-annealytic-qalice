"""
Balanced k-means QUBO pipeline with coarsening → QUBO build → anneal → feasibility repair → expand.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import dimod
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from qalice_qi.samplers import sample_bqm, SamplerConfig, default_sa_config, default_sqa_config

logger = logging.getLogger(__name__)


def coarsen_fit(
    X_train: pd.DataFrame, M: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MiniBatchKMeans]:
    """Fit MiniBatchKMeans to create M micro-clusters."""
    logger.info(f"Coarsening {len(X_train)} samples into {M} micro-clusters")
    
    model = MiniBatchKMeans(n_clusters=M, random_state=random_state, n_init=3)
    labels = model.fit_predict(X_train.values)
    centroids = model.cluster_centers_
    
    # Calculate weights (sizes) for each micro-cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    weights = np.zeros(M)
    weights[unique_labels] = counts
    
    logger.info(f"Created {M} micro-clusters with sizes: min={weights.min()}, max={weights.max()}, mean={weights.mean():.1f}")
    return centroids, labels, weights, model


def build_bqm(
    centroids: np.ndarray, 
    weights: np.ndarray, 
    k: int, 
    knn: int, 
    lambda1: float, 
    lambda2: float, 
    random_state: int
) -> dimod.BinaryQuadraticModel:
    """Build Binary Quadratic Model for balanced k-means QUBO."""
    M = len(centroids)
    W = weights.sum()
    
    logger.info(f"Building BQM: M={M}, k={k}, knn={knn}, W={W}")
    
    # Build KNN graph for sparsification
    nn = NearestNeighbors(n_neighbors=min(knn, M-1))
    nn.fit(centroids)
    distances, indices = nn.kneighbors(centroids)
    
    # Scale distances to [0,1]
    max_dist = distances.max()
    if max_dist > 0:
        distances = distances / max_dist
    
    bqm = dimod.BinaryQuadraticModel('BINARY')
    
    # Add variables x_{i,k}
    for i in range(M):
        for k_idx in range(k):
            bqm.add_variable(f'x_{i}_{k_idx}')
    
    # Objective: sum_k sum_{i<j} w_i w_j d_ij x_{i,k} x_{j,k}
    for i in range(M):
        for neighbor_idx in range(1, len(indices[i])):  # Skip self (index 0)
            j = indices[i][neighbor_idx]
            if i < j:  # Ensure i < j to avoid double counting
                d_ij = distances[i][neighbor_idx]
                coeff = weights[i] * weights[j] * d_ij
                
                for k_idx in range(k):
                    bqm.add_interaction(f'x_{i}_{k_idx}', f'x_{j}_{k_idx}', coeff)
    
    # Constraint 1: Each micro-cluster assigned to exactly one macro cluster
    # lambda1 * sum_i (sum_k x_{i,k} - 1)^2
    for i in range(M):
        # Linear terms: -2 * lambda1 * sum_k x_{i,k}
        for k_idx in range(k):
            bqm.add_variable(f'x_{i}_{k_idx}', -2 * lambda1)
        
        # Quadratic terms: lambda1 * sum_k sum_l x_{i,k} x_{i,l}
        for k1 in range(k):
            for k2 in range(k):
                if k1 == k2:
                    bqm.add_variable(f'x_{i}_{k1}', lambda1)
                else:
                    bqm.add_interaction(f'x_{i}_{k1}', f'x_{i}_{k2}', lambda1)
    
    # Constraint 2: Balanced clusters
    # lambda2 * sum_k (sum_i w_i x_{i,k} - W/K)^2
    target_weight = W / k
    for k_idx in range(k):
        # Linear terms: -2 * lambda2 * target_weight * sum_i w_i x_{i,k}
        for i in range(M):
            bqm.add_variable(f'x_{i}_{k_idx}', -2 * lambda2 * target_weight * weights[i])
        
        # Quadratic terms: lambda2 * sum_i sum_j w_i w_j x_{i,k} x_{j,k}
        for i in range(M):
            for j in range(M):
                if i == j:
                    bqm.add_variable(f'x_{i}_{k_idx}', lambda2 * weights[i] * weights[j])
                else:
                    bqm.add_interaction(f'x_{i}_{k_idx}', f'x_{j}_{k_idx}', lambda2 * weights[i] * weights[j])
    
    # Add constant term for constraint 2
    bqm.offset += lambda2 * k * (target_weight ** 2)
    
    logger.info(f"BQM built: {len(bqm.variables)} variables, {len(bqm.quadratic)} couplers")
    return bqm





def decode_assignments(
    sampleset: dimod.SampleSet, 
    M: int, 
    k: int, 
    weights: np.ndarray, 
    W: float, 
    balance_tol_frac: float = 0.05
) -> np.ndarray:
    """Decode sample to cluster assignments with feasibility repair."""
    logger.info("Decoding assignments and performing feasibility repair")
    
    best_sample = sampleset.first.sample
    
    # Decode one-hot per micro-cluster
    assignments = np.zeros(M, dtype=int)
    
    for i in range(M):
        # Get probabilities for each cluster
        probs = np.zeros(k)
        for k_idx in range(k):
            var_name = f'x_{i}_{k_idx}'
            if var_name in best_sample:
                probs[k_idx] = best_sample[var_name]
        
        # Assign to cluster with highest probability
        assignments[i] = np.argmax(probs)
    
    # Balance repair
    target_weight = W / k
    tolerance = balance_tol_frac * target_weight
    
    for iteration in range(10):  # Max 10 repair iterations
        cluster_weights = np.zeros(k)
        for cluster_id in range(k):
            mask = assignments == cluster_id
            cluster_weights[cluster_id] = weights[mask].sum()
        
        # Check if balanced
        imbalances = np.abs(cluster_weights - target_weight)
        if np.all(imbalances <= tolerance):
            break
        
        # Find most imbalanced clusters
        over_idx = np.argmax(cluster_weights - target_weight)
        under_idx = np.argmin(cluster_weights - target_weight)
        
        if cluster_weights[over_idx] <= target_weight + tolerance:
            break
        
        # Find micro-cluster to reassign (smallest weight in over-cluster)
        over_mask = assignments == over_idx
        if not np.any(over_mask):
            break
            
        over_indices = np.where(over_mask)[0]
        lightest_idx = over_indices[np.argmin(weights[over_indices])]
        
        # Reassign to under-cluster
        assignments[lightest_idx] = under_idx
        logger.debug(f"Repair iteration {iteration+1}: moved micro {lightest_idx} from cluster {over_idx} to {under_idx}")
    
    # Final balance check
    final_weights = np.zeros(k)
    for cluster_id in range(k):
        mask = assignments == cluster_id
        final_weights[cluster_id] = weights[mask].sum()
    
    violations = np.abs(final_weights - target_weight) / target_weight
    logger.info(f"Final balance violations: max={violations.max():.3f}, mean={violations.mean():.3f}")
    
    return assignments


def expand_to_rows(micro_labels: np.ndarray, micro_of_row: np.ndarray) -> np.ndarray:
    """Expand micro-cluster assignments to original rows."""
    return micro_labels[micro_of_row]


def run_qubo(
    prefix: str,
    k: int = 8,
    M: int = 400,
    knn: int = 40,
    lambda1: float = 5.0,
    lambda2: float = 1.0,
    sweeps: int = 10_000,
    num_reads: int = 64,
    random_state: int = 42,
    sampler: str = "sa"
) -> Dict[str, Any]:
    """Run complete QUBO balanced k-means pipeline."""
    
    logger.info(f"Starting QUBO pipeline for prefix='{prefix}', k={k}, M={M}, sampler={sampler}")
    
    # Setup paths
    base_path = Path("data/processed/bank_marketing")
    reports_path = Path("reports/qi") / prefix / sampler
    reports_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
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
    
    # Create modeling matrix
    modeling_features = schema["one_hot_features"] + schema["scaler_features"]
    X = df[modeling_features]
    
    # Split by month_idx
    train_mask = df["month_idx"].isin(splits["train_months"])
    valid_mask = df["month_idx"].isin(splits["valid_months"])
    
    X_train = X[train_mask]
    X_valid = X[valid_mask]
    
    logger.info(f"Train: {len(X_train)} samples, Valid: {len(X_valid)} samples")
    
    # Timing
    timings = {}
    total_start = time.time()
    
    # 1. Coarsen
    start_time = time.time()
    centroids, micro_labels_train, weights, coarsen_model = coarsen_fit(X_train, M, random_state)
    timings["coarsen_sec"] = time.time() - start_time
    
    # 2. Build BQM
    start_time = time.time()
    bqm = build_bqm(centroids, weights, k, knn, lambda1, lambda2, random_state)
    timings["bqm_build_sec"] = time.time() - start_time
    
    # 3. Sample BQM
    start_time = time.time()
    cfg = default_sa_config(random_state) if sampler == "sa" else default_sqa_config(random_state)
    cfg["sweeps"] = sweeps
    cfg["num_reads"] = num_reads
    sampleset = sample_bqm(bqm, cfg)
    timings["solve_sec"] = time.time() - start_time
    
    # 4. Decode
    start_time = time.time()
    W = weights.sum()
    macro_assignments = decode_assignments(sampleset, M, k, weights, W)
    timings["decode_sec"] = time.time() - start_time
    
    # 5. Expand
    start_time = time.time()
    
    # Expand train
    train_assignments = expand_to_rows(macro_assignments, micro_labels_train)
    
    # Expand valid (predict micro-clusters first)
    valid_micro_labels = coarsen_model.predict(X_valid.values)
    valid_assignments = expand_to_rows(macro_assignments, valid_micro_labels)
    
    timings["expand_sec"] = time.time() - start_time
    timings["total_sec"] = time.time() - total_start
    
    # Save artifacts
    config = {
        "k": k, "M": M, "knn": knn, "lambda1": lambda1, "lambda2": lambda2,
        "sweeps": sweeps, "reads": num_reads, "prefix": prefix, "sampler": sampler,
        "sampler_config": dict(cfg)
    }
    
    bqm_stats = {
        "n_variables": len(bqm.variables),
        "n_couplers": len(bqm.quadratic),
        "build_sec": timings["bqm_build_sec"],
        "solve_sec": timings["solve_sec"],
        "best_energy": float(sampleset.first.energy),
        "violations": {}
    }
    
    # Calculate violations
    final_weights = np.zeros(k)
    for cluster_id in range(k):
        mask = macro_assignments == cluster_id
        final_weights[cluster_id] = weights[mask].sum()
    
    target_weight = W / k
    for cluster_id in range(k):
        bqm_stats["violations"][f"cluster_{cluster_id}"] = float(abs(final_weights[cluster_id] - target_weight) / target_weight)
    
    # Save files
    with open(reports_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    with open(reports_path / "bqm_stats.json", "w") as f:
        json.dump(bqm_stats, f, indent=2)
    
    with open(reports_path / "timings.json", "w") as f:
        json.dump(timings, f, indent=2)
    
    # Save assignments
    train_df = pd.DataFrame({
        "index": df[train_mask].index,
        "cluster_id": train_assignments,
        "split": "train"
    })
    train_df.to_parquet(reports_path / "assignments_train.parquet", index=False)
    
    valid_df = pd.DataFrame({
        "index": df[valid_mask].index,
        "cluster_id": valid_assignments,
        "split": "valid"
    })
    valid_df.to_parquet(reports_path / "assignments_valid.parquet", index=False)
    
    # Save micro mapping
    micro_map_data = []
    for row_idx, micro_id in enumerate(micro_labels_train):
        micro_map_data.append({
            "row_index": df[train_mask].index[row_idx],
            "micro_id": micro_id,
            "centroid_id": micro_id,
            "weight": 1
        })
    
    # Add micro cluster info
    for micro_id in range(M):
        micro_map_data.append({
            "row_index": -1,  # Sentinel for micro cluster info
            "micro_id": micro_id,
            "centroid_id": micro_id,
            "weight": int(weights[micro_id])
        })
    
    micro_df = pd.DataFrame(micro_map_data)
    micro_df.to_parquet(reports_path / "micro_map.parquet", index=False)
    
    logger.info(f"QUBO pipeline complete. Results saved to {reports_path}")
    
    return {
        "config": config,
        "bqm_stats": bqm_stats,
        "timings": timings,
        "train_assignments": train_assignments,
        "valid_assignments": valid_assignments,
        "out_dir": str(reports_path),
        "sampler": sampler
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run for both prefixes and samplers
    for prefix in ["with_sentiment", "no_sentiment"]:
        for sampler in ["sa", "sqa"]:
            try:
                result = run_qubo(prefix, sampler=sampler)
                print(f"✓ Completed QUBO pipeline for {prefix} with {sampler}")
            except Exception as e:
                print(f"✗ Failed for {prefix} with {sampler}: {e}")