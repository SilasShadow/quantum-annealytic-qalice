"""CLI for qAlice_core pipeline stages."""

import logging
import sys
import time
from enum import Enum
from pathlib import Path

import pandas as pd
import typer
from typing_extensions import Annotated

from .data_ingest import load_bank_marketing
from .feature_engineering import build_feature_view, persist_feature_view
from .splits import temporal_holdout
from .toggles import fv_path
from . import baselines, propensity

try:
    from qlattice_viz import visuals
except ImportError:
    visuals = None

# Add project root to path for development
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from qalice_qi import qubo_balanced_kmeans, runner, compare
except ImportError:
    qubo_balanced_kmeans = None
    runner = None
    compare = None

try:
    from qalice_viz import runtime_sentiment
except ImportError:
    runtime_sentiment = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Lattice Core Pipeline CLI")
qi_app = typer.Typer(help="Quantum-inspired QUBO clustering commands")
app.add_typer(qi_app, name="qi")


class Mode(str, Enum):
    """Feature engineering modes."""

    both = "both"
    on = "on"
    off = "off"


@app.command()
def stage1(
    mode: Annotated[Mode, typer.Option(help="Feature engineering mode")] = Mode.both,
) -> None:
    """Run Stage-1 end-to-end pipeline: ingest -> feature engineering -> splits."""
    try:
        logger.info("Starting Stage-1 pipeline")

        # Step 1: Load and clean data
        logger.info("Step 1: Loading and cleaning bank marketing data")
        df = load_bank_marketing()
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

        # Step 2: Build feature views based on mode
        modes_to_run = []
        if mode == Mode.both:
            modes_to_run = [True, False]
        elif mode == Mode.on:
            modes_to_run = [True]
        elif mode == Mode.off:
            modes_to_run = [False]

        logger.info(f"Step 2: Building feature views for modes: {modes_to_run}")

        for include_sentiment in modes_to_run:
            mode_name = "with_sentiment" if include_sentiment else "no_sentiment"
            logger.info(f"Building {mode_name} feature view")

            fv, schema, scaler = build_feature_view(df, include_sentiment=include_sentiment)
            persist_feature_view(
                fv, schema, include_sentiment=include_sentiment, fitted_scaler=scaler
            )

            logger.info(
                f"Features included in {mode_name}: "
                f"numeric={len(schema['numeric_features'])}, "
                f"one_hot={len(schema['one_hot_features'])}"
            )

        # Step 3: Create temporal splits using with_sentiment view
        logger.info("Step 3: Creating temporal splits")

        # Use with_sentiment view if available, otherwise use the built view
        if True in modes_to_run:
            fv_with = pd.read_parquet(fv_path(include_sentiment=True))
        else:
            fv_with = pd.read_parquet(fv_path(include_sentiment=False))

        train, valid = temporal_holdout(fv_with, valid_fraction=0.2)

        logger.info("Stage-1 pipeline completed successfully")

        # Log output inventory
        logger.info("Output inventory:")
        output_files = [
            "data/interim/bank_marketing_clean.csv",
            "data/interim/bank_marketing_provenance.json",
            "data/processed/bank_marketing/splits.json",
        ]

        for include_sentiment in modes_to_run:
            prefix = "with_sentiment" if include_sentiment else "no_sentiment"
            output_files.extend(
                [
                    f"data/processed/bank_marketing/scaler_{prefix}.pkl",
                    f"data/processed/bank_marketing/{prefix}_feature_view.parquet",
                    f"data/processed/bank_marketing/{prefix}_feature_schema.json",
                ]
            )

        for file_path in output_files:
            if Path(file_path).exists():
                logger.info(f"✓ {file_path}")
            else:
                logger.warning(f"✗ {file_path} (missing)")

    except Exception as e:
        logger.error(f"Stage-1 pipeline failed: {e}")
        sys.exit(1)


@app.command()
def baselines(
    k_list: Annotated[str, typer.Option(help="Comma-separated k values")] = "8",
    algo: Annotated[str, typer.Option(help="Algorithm: kmeans, gmm, or both")] = "both",
    prefix: Annotated[str, typer.Option(help="Prefix: with_sentiment, no_sentiment, or both")] = "both",
) -> None:
    """Run clustering baselines and propensity modeling."""
    try:
        # Parse k_list
        k_values = [int(k.strip()) for k in k_list.split(",")]
        
        # Determine prefixes to run
        if prefix == "both":
            prefixes = ["with_sentiment", "no_sentiment"]
        elif prefix in ["with_sentiment", "no_sentiment"]:
            prefixes = [prefix]
        else:
            raise ValueError(f"Invalid prefix: {prefix}")
        
        # Validate algo
        if algo not in ["kmeans", "gmm", "both"]:
            raise ValueError(f"Invalid algo: {algo}")
        
        logger.info(f"Running baselines with k={k_values}, algo={algo}, prefixes={prefixes}")
        
        # Step 1: Run clustering for each prefix
        for pref in prefixes:
            logger.info(f"Running clustering for {pref}")
            baselines.run_clustering(pref, k_list=k_values)
        
        # Step 2: Train propensity models
        for pref in prefixes:
            logger.info(f"Training propensity model for {pref}")
            propensity.train_and_evaluate(pref)
        
        # Step 3: Build visuals (if available)
        if visuals:
            first_k = k_values[0]
            
            for pref in prefixes:
                # Determine which algo to use for visuals
                visual_algo = "kmeans" if algo in ["both", "kmeans"] else "gmm"
                
                logger.info(f"Building KPI bars for {pref}")
                visuals.build_kpi_bars(prefix=pref, algo=visual_algo, k=first_k)
            
            logger.info("Building comparison summary")
            visual_algo = "kmeans" if algo in ["both", "kmeans"] else "gmm"
            visuals.build_comparison_summary(k=first_k, algo=visual_algo)
        else:
            logger.warning("Visuals module not available, skipping visualization step")
        
        # Step 4: Print summary for each prefix
        for pref in prefixes:
            try:
                # Load metrics
                first_k = k_values[0]
                summary_algo = "kmeans" if algo in ["both", "kmeans"] else "gmm"
                
                # Load clustering metrics
                metrics_path = Path(f"reports/baselines/{pref}/{summary_algo}_k{first_k}_metrics.json")
                silhouette = None
                if metrics_path.exists():
                    import json
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    silhouette = metrics.get("silhouette")
                
                # Load propensity metrics
                prop_path = Path(f"reports/propensity/{pref}/metrics.json")
                auc = None
                if prop_path.exists():
                    import json
                    with open(prop_path) as f:
                        prop_metrics = json.load(f)
                    auc = prop_metrics.get("auc")
                
                # Load top decile lift
                decile_path = Path(f"reports/propensity/{pref}/decile_lift.csv")
                top_lift = None
                if decile_path.exists():
                    decile_df = pd.read_csv(decile_path)
                    top_lift = decile_df[decile_df["decile"] == 1]["lift_vs_overall"].iloc[0]
                
                # Format values
                sil_str = f"{silhouette:.3f}" if silhouette is not None else "N/A"
                auc_str = f"{auc:.3f}" if auc is not None else "N/A"
                lift_str = f"{top_lift:.2f}" if top_lift is not None else "N/A"
                
                print(f"[{pref}] silhouette={sil_str}, AUC={auc_str}, top-decile lift={lift_str}")
                
            except Exception as e:
                logger.warning(f"Could not generate summary for {pref}: {e}")
                print(f"[{pref}] metrics unavailable")
        
        logger.info("Baselines pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Baselines pipeline failed: {e}")
        sys.exit(1)


@qi_app.command()
def run(
    prefix: Annotated[str, typer.Option(help="Feature prefix: with_sentiment or no_sentiment")] = "with_sentiment",
    k: Annotated[int, typer.Option(help="Number of clusters")] = 8,
    M: Annotated[int, typer.Option(help="Number of micro-clusters")] = 400,
    knn: Annotated[int, typer.Option(help="KNN neighbors for sparsification")] = 40,
    lambda1: Annotated[float, typer.Option(help="Assignment constraint weight")] = 5.0,
    lambda2: Annotated[float, typer.Option(help="Balance constraint weight")] = 1.0,
    sweeps: Annotated[int, typer.Option(help="Annealing sweeps")] = 10000,
    reads: Annotated[int, typer.Option(help="Number of reads")] = 64,
    sampler: Annotated[str, typer.Option(help="Sampler type: sa or sqa")] = "sa",
) -> None:
    """Run QUBO balanced k-means clustering."""
    if qubo_balanced_kmeans is None:
        logger.error("qalice_qi module not available")
        sys.exit(1)
    
    if sampler not in ["sa", "sqa"]:
        logger.error(f"Invalid sampler: {sampler}. Must be 'sa' or 'sqa'")
        sys.exit(1)
    
    try:
        logger.info(f"Running QUBO clustering: prefix={prefix}, k={k}, M={M}, sampler={sampler}")
        
        start_time = time.time()
        result = qubo_balanced_kmeans.run_qubo(
            prefix=prefix,
            k=k,
            M=M,
            knn=knn,
            lambda1=lambda1,
            lambda2=lambda2,
            sweeps=sweeps,
            num_reads=reads,
            sampler=sampler
        )
        total_time = time.time() - start_time
        
        # Print required format
        timings = result["timings"]
        print(f"QI[{prefix}] {sampler} total_sec={timings['total_sec']:.2f}, solve_sec={timings['solve_sec']:.2f}, out_dir={result['out_dir']}")
        
        logger.info(f"QUBO clustering completed. Artifacts saved to {result['out_dir']}")
        
    except Exception as e:
        logger.error(f"QUBO clustering failed: {e}")
        sys.exit(1)


@qi_app.command()
def scan(
    prefix: Annotated[str, typer.Option(help="Feature prefix: with_sentiment or no_sentiment")] = "with_sentiment",
    k: Annotated[int, typer.Option(help="Number of clusters")] = 8,
    sampler: Annotated[str, typer.Option(help="Sampler type: sa or sqa")] = "sa",
) -> None:
    """Scan hyperparameters and select best configuration."""
    if runner is None:
        logger.error("qalice_qi module not available")
        sys.exit(1)
    
    if sampler not in ["sa", "sqa"]:
        logger.error(f"Invalid sampler: {sampler}. Must be 'sa' or 'sqa'")
        sys.exit(1)
    
    try:
        logger.info(f"Running hyperparameter scan: prefix={prefix}, k={k}, sampler={sampler}")
        
        start_time = time.time()
        result = runner.scan_and_select(prefix=prefix, k=k, sampler=sampler)
        total_time = time.time() - start_time
        
        # Print required format
        best = result["best_params"]
        selected = result["selected_result"]
        
        print(f"Best ({sampler}) λ1={best['lambda1']}, λ2={best['lambda2']}, silhouette={selected['silhouette_valid']:.4f}, total_sec={total_time:.2f}")
        
        logger.info(f"Scan completed. Results saved to reports/qi/{prefix}/")
        
    except Exception as e:
        logger.error(f"Hyperparameter scan failed: {e}")
        sys.exit(1)


@qi_app.command()
def compare(
    prefix: Annotated[str, typer.Option(help="Feature prefix: with_sentiment or no_sentiment")] = "with_sentiment",
    algo: Annotated[str, typer.Option(help="Baseline algorithm")] = "kmeans",
    k: Annotated[int, typer.Option(help="Number of clusters")] = 8,
) -> None:
    """Compare QUBO clusters with baseline and generate visualizations."""
    if compare is None:
        logger.error("qalice_qi.compare module not available")
        sys.exit(1)
    
    try:
        logger.info(f"Running comparison: prefix={prefix}, algo={algo}, k={k}")
        
        # Step 1: Compare with baseline
        result = compare.compare_with_baseline(prefix=prefix, algo=algo, k=k)
        
        overlap_png = f"reports/qi/{prefix}/overlap_{algo}_k{k}.png"
        print(f"Overlap (NMI={result['nmi']:.3f}, ARI={result['ari']:.3f}) saved to {overlap_png}")
        
        # Step 2: Build runtime chart
        if runtime_sentiment is not None:
            runtime_png = runtime_sentiment.build_runtime_bar(prefix=prefix)
            print(f"Runtime chart saved to {runtime_png}")
            
            # Step 3: Build sentiment toggle summary
            summary_df = runtime_sentiment.build_sentiment_toggle_summary(algo=algo, k=k)
            print("\nSentiment Toggle Summary:")
            print(summary_df.to_string(index=False))
        else:
            logger.warning("qalice_viz.runtime_sentiment module not available, skipping visualizations")
        
        logger.info("Comparison completed successfully")
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
