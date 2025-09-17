"""
CLI interface for QUBO balanced k-means pipeline.
"""

import argparse
import logging
import sys
from typing import Optional

from .qubo_balanced_kmeans import run_qubo


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main(args: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run QUBO balanced k-means clustering pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--prefix", 
        choices=["with_sentiment", "no_sentiment", "both"],
        default="both",
        help="Feature set prefix to process"
    )
    
    parser.add_argument(
        "--k", 
        type=int, 
        default=8,
        help="Number of clusters"
    )
    
    parser.add_argument(
        "--M", 
        type=int, 
        default=400,
        help="Number of micro-clusters for coarsening"
    )
    
    parser.add_argument(
        "--knn", 
        type=int, 
        default=40,
        help="KNN neighbors for sparsification"
    )
    
    parser.add_argument(
        "--lambda1", 
        type=float, 
        default=5.0,
        help="Assignment constraint weight"
    )
    
    parser.add_argument(
        "--lambda2", 
        type=float, 
        default=1.0,
        help="Balance constraint weight"
    )
    
    parser.add_argument(
        "--sweeps", 
        type=int, 
        default=10_000,
        help="Number of annealing sweeps"
    )
    
    parser.add_argument(
        "--num-reads", 
        type=int, 
        default=64,
        help="Number of annealing reads"
    )
    
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parsed_args = parser.parse_args(args)
    setup_logging(parsed_args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Determine prefixes to process
    if parsed_args.prefix == "both":
        prefixes = ["with_sentiment", "no_sentiment"]
    else:
        prefixes = [parsed_args.prefix]
    
    success_count = 0
    
    for prefix in prefixes:
        logger.info(f"Starting QUBO pipeline for prefix: {prefix}")
        
        try:
            result = run_qubo(
                prefix=prefix,
                k=parsed_args.k,
                M=parsed_args.M,
                knn=parsed_args.knn,
                lambda1=parsed_args.lambda1,
                lambda2=parsed_args.lambda2,
                sweeps=parsed_args.sweeps,
                num_reads=parsed_args.num_reads,
                random_state=parsed_args.random_state
            )
            
            logger.info(f"✅ Successfully completed QUBO pipeline for {prefix}")
            logger.info(f"   Total time: {result['timings']['total_sec']:.2f}s")
            logger.info(f"   Best energy: {result['bqm_stats']['best_energy']:.6f}")
            logger.info(f"   Variables: {result['bqm_stats']['n_variables']}")
            logger.info(f"   Couplers: {result['bqm_stats']['n_couplers']}")
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to process {prefix}: {e}")
            if parsed_args.verbose:
                logger.exception("Full traceback:")
    
    if success_count == len(prefixes):
        logger.info(f"🎉 All {len(prefixes)} pipeline(s) completed successfully")
        return 0
    else:
        logger.error(f"⚠️  Only {success_count}/{len(prefixes)} pipeline(s) succeeded")
        return 1


if __name__ == "__main__":
    sys.exit(main())