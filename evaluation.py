"""
Model evaluation utilities.

Computes RMSE, MAE, MAPE on validation sets and builds a comparison table
to pick the best model per state.
"""

import numpy as np
import pandas as pd
from src.logger import get_logger

logger = get_logger("evaluation")


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.mean(np.abs(actual - predicted))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MAPE — skip zeros in actual to avoid div by zero."""
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def evaluate_model(actual: np.ndarray, predicted: np.ndarray, model_name: str) -> dict:
    """Compute all metrics for a single model's predictions."""
    return {
        "model": model_name,
        "rmse": round(rmse(actual, predicted), 2),
        "mae": round(mae(actual, predicted), 2),
        "mape": round(mape(actual, predicted), 2),
    }


def build_comparison_table(results: list) -> pd.DataFrame:
    """
    results: list of dicts from evaluate_model
    Returns a sorted DataFrame with all models' metrics.
    """
    df = pd.DataFrame(results)
    # rank models by MAPE (lower is better), use RMSE as tiebreaker
    df = df.sort_values(["mape", "rmse"]).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def select_best_model(comparison_df: pd.DataFrame) -> str:
    """
    Pick the model with lowest MAPE.
    Simple rule: if MAPE differs by < 1%, prefer simpler model (SARIMA/Prophet over XGB/LSTM).
    """
    if comparison_df.empty:
        return "sarima"  # fallback

    best = comparison_df.iloc[0]["model"]
    best_mape = comparison_df.iloc[0]["mape"]

    # if within 1% MAPE, prefer interpretable models
    simple_models = ["sarima", "prophet"]
    for _, row in comparison_df.iterrows():
        if row["model"] in simple_models and (row["mape"] - best_mape) < 1.0:
            logger.info(
                f"Preferring {row['model']} over {best} "
                f"(MAPE diff: {row['mape'] - best_mape:.2f}%)"
            )
            return row["model"]

    logger.info(f"Best model by MAPE: {best} ({best_mape:.2f}%)")
    return best


def log_state_results(state: str, comparison_df: pd.DataFrame, best_model: str):
    logger.info(f"\n{'='*50}")
    logger.info(f"  Results for: {state}")
    logger.info(f"{'='*50}")
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    logger.info(f"\n  >> Best model: {best_model}")
    logger.info(f"{'='*50}\n")
