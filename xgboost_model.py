"""
XGBoost forecasting model.

Treats time series forecasting as supervised regression using lag features.
For multi-step ahead forecasting we use a recursive strategy — 
predict 1 step, append to history, predict next, etc.
This accumulates error over long horizons but works reasonably for 8 weeks.
"""

import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from src.feature_engineering import (
    build_features,
    get_feature_columns,
    time_series_split,
)
from src.logger import get_logger

logger = get_logger("xgboost_model")


def fit_xgboost(train_df: pd.DataFrame, state: str) -> dict:
    """
    Build lag features on train data and fit XGBoost.
    train_df: DataFrame with 'date' and 'sales'
    """
    logger.info(f"[{state}] Building features for XGBoost...")

    feat_df = build_features(train_df)
    feature_cols = get_feature_columns()

    # check we actually have all columns
    available = [c for c in feature_cols if c in feat_df.columns]
    if len(available) < len(feature_cols):
        missing = set(feature_cols) - set(available)
        logger.warning(f"[{state}] Missing feature columns: {missing}")

    X = feat_df[available].values
    y = feat_df["sales"].values

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y, verbose=False)

    # quick in-sample check
    train_preds = model.predict(X)
    train_rmse = np.sqrt(np.mean((y - train_preds) ** 2))
    logger.info(f"[{state}] XGBoost train RMSE: {train_rmse:,.0f}")

    return {
        "model": model,
        "feature_cols": available,
        "train_data": train_df.copy(),  # needed for recursive forecasting
    }


def forecast_xgboost(model_dict: dict, n_steps: int = 8) -> np.ndarray:
    """
    Recursive multi-step forecast.
    We append each prediction back to the history and re-compute features.
    """
    model = model_dict["model"]
    feature_cols = model_dict["feature_cols"]
    history_df = model_dict["train_data"].copy()

    preds = []
    current_df = history_df.copy()

    for step in range(n_steps):
        # build features on current history
        from src.feature_engineering import build_features, add_time_features, add_holiday_flag
        import holidays

        feat_df = build_features(current_df)
        if len(feat_df) == 0:
            logger.error("Feature building returned empty df — stopping recursion")
            break

        last_row = feat_df.iloc[[-1]]
        available_cols = [c for c in feature_cols if c in last_row.columns]
        X_step = last_row[available_cols].values

        pred = model.predict(X_step)[0]
        pred = max(0, pred)  # clip negatives
        preds.append(pred)

        # append prediction to history for next step
        next_date = current_df["date"].max() + pd.Timedelta(weeks=1)
        new_row = pd.DataFrame({"date": [next_date], "sales": [pred]})
        current_df = pd.concat([current_df, new_row], ignore_index=True)

    return np.array(preds)


def get_feature_importance(model_dict: dict) -> pd.DataFrame:
    """Returns sorted feature importance from XGBoost."""
    model = model_dict["model"]
    feature_cols = model_dict["feature_cols"]
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi_df


def save_xgboost(model_dict: dict, state: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"xgboost_{state.replace(' ', '_')}.pkl")
    # don't save training data in pkl — it bloats the file
    save_dict = {
        "model": model_dict["model"],
        "feature_cols": model_dict["feature_cols"],
        "train_data": model_dict["train_data"],
    }
    with open(path, "wb") as f:
        pickle.dump(save_dict, f)
    logger.info(f"[{state}] XGBoost saved to {path}")
    return path


def load_xgboost(state: str, save_dir: str) -> dict:
    path = os.path.join(save_dir, f"xgboost_{state.replace(' ', '_')}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
