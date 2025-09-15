"""CLI for qAlice_core pipeline stages."""

import logging
import sys
from enum import Enum
from pathlib import Path

import pandas as pd
import typer
from typing_extensions import Annotated

from .data_ingest import load_bank_marketing
from .feature_engineering import build_feature_view, persist_feature_view
from .splits import temporal_holdout
from .toggles import fv_path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Lattice Core Pipeline CLI")


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


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
