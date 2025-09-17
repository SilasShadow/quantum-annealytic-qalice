# qAlice Quantum Intelligence (qalice_qi)

Production-grade Balanced k-means QUBO pipeline for quantum annealing.

## Overview

This module implements a complete pipeline for balanced k-means clustering using Quantum Unconstrained Binary Optimization (QUBO):

```
Data → Coarsen → QUBO Build → Anneal → Feasibility Repair → Expand → Results
```

## Pipeline Stages

### 1. Coarsening
- Uses MiniBatchKMeans to create M micro-clusters (default M=400)
- Reduces problem size for quantum annealing
- Preserves data distribution and cluster structure

### 2. QUBO Formulation
- Variables: `x_{i,k} ∈ {0,1}` where i∈{1..M}, k∈{1..K}
- Objective: Minimize intra-cluster distances + constraint violations
- Constraints: Assignment (each micro to one macro) + Balance (equal cluster sizes)

### 3. Quantum Annealing
- Prefers `neal.SimulatedAnnealingSampler` (D-Wave Neal)
- Fallbacks: `dimod` reference samplers
- Configurable sweeps and reads for solution quality

### 4. Feasibility Repair
- Decodes one-hot assignments via argmax
- Enforces balance constraints through greedy reassignment
- Maintains solution quality while ensuring feasibility

### 5. Expansion
- Maps original data points to final cluster assignments
- Preserves temporal train/validation splits

## Usage

### Command Line Interface

```bash
# Install quantum dependencies
pip install -r requirements-optional.txt

# Run for both feature sets
python -m qalice_qi

# Run for specific feature set
python -m qalice_qi --prefix with_sentiment

# Custom parameters
python -m qalice_qi --k 12 --M 500 --sweeps 20000
```

### Python API

```python
from qalice_qi import run_qubo

# Run complete pipeline
result = run_qubo(
    prefix="with_sentiment",
    k=8,                    # Number of clusters
    M=400,                  # Number of micro-clusters  
    knn=40,                 # KNN sparsification
    lambda1=5.0,            # Assignment constraint weight
    lambda2=1.0,            # Balance constraint weight
    sweeps=10_000,          # Annealing sweeps
    num_reads=64,           # Annealing reads
    random_state=42         # Reproducibility seed
)
```

### Individual Functions

```python
from qalice_qi import coarsen_fit, build_bqm, anneal_bqm

# Coarsen data
centroids, labels, weights, model = coarsen_fit(X_train, M=400, random_state=42)

# Build QUBO
bqm = build_bqm(centroids, weights, k=8, knn=40, lambda1=5.0, lambda2=1.0, random_state=42)

# Solve with quantum annealing
sampleset = anneal_bqm(bqm, sweeps=10_000, num_reads=64, random_state=42)
```

## Input Requirements

### Data Files
- `data/processed/bank_marketing/{prefix}_feature_view.parquet`
- `data/processed/bank_marketing/{prefix}_feature_schema.json`  
- `data/processed/bank_marketing/splits.json`

### Schema Structure
```json
{
  "one_hot_features": ["job_admin.", "marital_single", ...],
  "scaler_features": ["age_scaled", "campaign_scaled", ...],
  "passthrough_features": ["y", "month_idx", ...]
}
```

## Output Artifacts

All outputs saved to `reports/qi/{prefix}/`:

### Configuration
- `config.json` - Pipeline parameters and settings

### Performance Metrics  
- `bqm_stats.json` - QUBO size, energy, violations
- `timings.json` - Stage-wise execution times

### Results
- `assignments_train.parquet` - Training set cluster assignments
- `assignments_valid.parquet` - Validation set cluster assignments  
- `micro_map.parquet` - Micro-cluster mapping and metadata

## QUBO Mathematical Formulation

### Objective Function
```
minimize: Σ_k Σ_{i<j} w_i w_j d_ij x_{i,k} x_{j,k}
        + λ₁ Σ_i (Σ_k x_{i,k} - 1)²  
        + λ₂ Σ_k (Σ_i w_i x_{i,k} - W/K)²
```

Where:
- `w_i` = weight (size) of micro-cluster i
- `d_ij` = Euclidean distance between micro-centroids i and j  
- `W` = total weight = Σ_i w_i
- `λ₁` = assignment constraint penalty
- `λ₂` = balance constraint penalty

### Performance Optimizations
- **KNN Sparsification**: Only include couplers for k-nearest neighbors
- **Distance Scaling**: Normalize distances to [0,1] range
- **Constraint Tuning**: Balance solution quality vs. constraint satisfaction

## Dependencies

### Core Requirements
```
pandas>=1.5.0
numpy>=1.24.0  
scikit-learn>=1.3.0
```

### Quantum Requirements (Optional)
```
dwave-ocean-sdk>=6.0.0  # Includes dimod, neal
# OR
dimod>=0.12.0
neal>=0.6.0
```

## Performance Characteristics

### Scalability
- **Input Size**: Handles 40K+ samples efficiently via coarsening
- **QUBO Size**: Typically M×K variables (400×8 = 3,200 variables)
- **Solve Time**: 10-60 seconds depending on parameters

### Quality Metrics
- **Balance Tolerance**: Default 5% deviation from equal cluster sizes
- **Energy Convergence**: Tracks best energy across multiple reads
- **Constraint Violations**: Reported per cluster in output stats

## Troubleshooting

### Common Issues

**Missing quantum dependencies:**
```bash
pip install -r requirements-optional.txt
```

**Memory issues with large datasets:**
- Reduce M (micro-clusters): `--M 200`
- Reduce KNN: `--knn 20`

**Poor solution quality:**
- Increase sweeps: `--sweeps 50000`
- Increase reads: `--num-reads 128`
- Tune constraint weights: `--lambda1 10.0 --lambda2 2.0`

**Imbalanced clusters:**
- Increase λ₂: `--lambda2 5.0`
- Check balance violations in `bqm_stats.json`

### Logging
Enable verbose logging for debugging:
```bash
python -m qalice_qi --verbose
```

## Integration

### With Existing Pipeline
The QUBO module integrates with the existing qAlice pipeline:

1. **Stage 1**: Feature engineering (`qalice_core.cli`)
2. **Stage 2**: QUBO clustering (`qalice_qi`)  
3. **Stage 3**: Evaluation and visualization (`qlattice_viz`)

### Custom Samplers
To use custom quantum hardware:

```python
from qalice_qi.qubo_balanced_kmeans import build_bqm
import dimod

# Build BQM
bqm = build_bqm(centroids, weights, k=8, knn=40, lambda1=5.0, lambda2=1.0, random_state=42)

# Use custom sampler
sampler = YourQuantumSampler()  # e.g., DWaveSampler
sampleset = sampler.sample(bqm, num_reads=100)
```

## References

- [D-Wave Ocean SDK](https://docs.ocean.dwavesys.com/)
- [QUBO Formulations](https://docs.dwavesys.com/docs/latest/c_handbook_3.html)
- [Balanced k-means Literature](https://link.springer.com/article/10.1007/s10994-018-5733-2)