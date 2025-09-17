"""Binary propensity modeling for bank marketing feature views."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    from sklearn.inspection import permutation_importance
    HAS_SHAP = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


def _load_feature_data(prefix: str) -> tuple[pd.DataFrame, dict, dict]:
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


def _prepare_modeling_data(df: pd.DataFrame, schema: dict, splits: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list]:
    """Prepare train/valid splits with modeling features and targets."""
    # Get modeling features (one_hot + numeric, excluding duration for no target leakage)
    modeling_features = schema["one_hot_features"] + schema["numeric_features"]
    modeling_features = [f for f in modeling_features if f != "duration" and f in df.columns]
    
    # Split by month buckets
    train_mask = df["month_idx"].isin(splits["train_months"])
    valid_mask = df["month_idx"].isin(splits["valid_months"])
    
    X_train = df[train_mask][modeling_features].copy()
    X_valid = df[valid_mask][modeling_features].copy()
    
    # Handle both string ("yes"/"no") and numeric (1/0) target encoding
    if df["y"].dtype == 'object':
        y_train = (df[train_mask]["y"] == "yes").astype(int)
        y_valid = (df[valid_mask]["y"] == "yes").astype(int)
    else:
        y_train = df[train_mask]["y"].astype(int)
        y_valid = df[valid_mask]["y"].astype(int)
    
    logger.info(f"Train: {len(X_train)} rows, {len(modeling_features)} features")
    logger.info(f"Valid: {len(X_valid)} rows, target rate: {y_valid.mean():.3f}")
    
    return X_train, X_valid, y_train, y_valid, modeling_features


def _train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Train XGBoost or fallback to GradientBoosting."""
    if HAS_XGBOOST:
        model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            random_state=42
        )
        logger.info("Training XGBoostClassifier")
    else:
        model = GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )
        logger.info("Training GradientBoostingClassifier (XGBoost not available)")
    
    model.fit(X_train, y_train)
    return model


def _evaluate_model(model: Any, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict:
    """Evaluate model on validation set."""
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    
    metrics = {
        "auc": roc_auc_score(y_valid, y_pred_proba),
        "pr_auc": average_precision_score(y_valid, y_pred_proba),
        "logloss": log_loss(y_valid, y_pred_proba),
        "brier": brier_score_loss(y_valid, y_pred_proba),
        "n_train": len(model.feature_importances_) if hasattr(model, 'feature_importances_') else 0,
        "n_valid": len(y_valid)
    }
    
    logger.info(f"AUC: {metrics['auc']:.3f}, PR-AUC: {metrics['pr_auc']:.3f}")
    logger.info(f"LogLoss: {metrics['logloss']:.3f}, Brier: {metrics['brier']:.3f}")
    
    return metrics


def _build_decile_lift_table(model: Any, X_valid: pd.DataFrame, y_valid: pd.Series) -> pd.DataFrame:
    """Build decile lift table on validation set."""
    scores = model.predict_proba(X_valid)[:, 1]
    overall_rate = y_valid.mean()
    
    # Create DataFrame and sort by score descending
    df = pd.DataFrame({
        "score": scores,
        "y": y_valid
    }).sort_values("score", ascending=False).reset_index(drop=True)
    
    # Split into 10 equal-count deciles
    n_per_decile = len(df) // 10
    deciles = []
    
    for i in range(10):
        start_idx = i * n_per_decile
        end_idx = (i + 1) * n_per_decile if i < 9 else len(df)
        
        decile_data = df.iloc[start_idx:end_idx]
        conversion_rate = decile_data["y"].mean()
        lift_vs_overall = conversion_rate / overall_rate if overall_rate > 0 else 0
        
        deciles.append({
            "decile": i + 1,
            "n": len(decile_data),
            "score_threshold": decile_data["score"].min(),
            "conversion_rate": conversion_rate,
            "lift_vs_overall": lift_vs_overall
        })
    
    return pd.DataFrame(deciles)


def _generate_feature_importance(model: Any, X_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.Series, output_dir: Path) -> None:
    """Generate SHAP summary or permutation importance."""
    if HAS_SHAP and HAS_MATPLOTLIB:
        try:
            # Use TreeExplainer for tree-based models
            explainer = shap.TreeExplainer(model)
            
            # Sample for SHAP (max 5k rows)
            sample_size = min(5000, len(X_train))
            sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
            X_sample = X_train.iloc[sample_idx]
            
            shap_values = explainer.shap_values(X_sample)
            
            # For binary classification, use positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Create summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.tight_layout()
            
            shap_path = output_dir / "shap_summary.png"
            plt.savefig(shap_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Saved SHAP summary: {shap_path}")
            return
            
        except Exception as e:
            logger.warning(f"SHAP failed: {e}, falling back to permutation importance")
    
    # Fallback to permutation importance
    perm_importance = permutation_importance(
        model, X_valid, y_valid, n_repeats=5, random_state=42
    )
    
    importance_df = pd.DataFrame({
        "feature": X_valid.columns,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std
    }).sort_values("importance_mean", ascending=False)
    
    importance_path = output_dir / "features_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Saved permutation importance: {importance_path}")


def train_and_evaluate(prefix: str) -> Dict[str, Any]:
    """Train and evaluate binary propensity model.
    
    Args:
        prefix: Either "with_sentiment" or "no_sentiment"
        
    Returns:
        Dictionary with evaluation metrics
    """
    if prefix not in ["with_sentiment", "no_sentiment"]:
        raise ValueError(f"Invalid prefix: {prefix}")
    
    # Load data
    df, schema, splits = _load_feature_data(prefix)
    X_train, X_valid, y_train, y_valid, modeling_features = _prepare_modeling_data(df, schema, splits)
    
    # Create output directory
    output_dir = Path(f"reports/propensity/{prefix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train model
    model = _train_model(X_train, y_train)
    
    # Evaluate model
    metrics = _evaluate_model(model, X_valid, y_valid)
    metrics["n_train"] = len(X_train)
    
    # Build decile lift table
    decile_lift = _build_decile_lift_table(model, X_valid, y_valid)
    
    # Generate feature importance
    _generate_feature_importance(model, X_train, X_valid, y_valid, output_dir)
    
    # Save artifacts
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    decile_lift.to_csv(output_dir / "decile_lift.csv", index=False)
    
    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(output_dir / "features_used.json", "w") as f:
        json.dump(modeling_features, f, indent=2)
    
    logger.info(f"Saved all artifacts to {output_dir}")
    
    return metrics