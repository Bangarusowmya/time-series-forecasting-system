"""
Main training pipeline.

For each state:
  1. Split data into train / validation
  2. Train all 4 models
  3. Evaluate on validation set
  4. Pick best model
  5. Save all models + results
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional

from src.data_preprocessing import preprocess_all_states
from src.feature_engineering import time_series_split
from src.evaluation import evaluate_model, build_comparison_table, select_best_model, log_state_results
from src.visualization import plot_forecast_vs_actual, plot_model_comparison, plot_feature_importance
from src.logger import get_logger

logger = get_logger("train_pipeline")

# where to save trained models
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def train_one_state(
    state: str,
    state_df: pd.DataFrame,
    models_dir: str = MODELS_DIR,
    outputs_dir: str = OUTPUTS_DIR,
    skip_lstm: bool = False,
) -> dict:
    """
    Full training loop for a single state.
    Returns dict with best model name, metrics, and future predictions.
    skip_lstm: useful for quick testing since LSTM is slowest to train
    """
    logger.info(f"\n{'#'*60}")
    logger.info(f"  Training: {state}")
    logger.info(f"{'#'*60}")

    # time-based train/val split — last 16 weeks for validation
    train_df, val_df = time_series_split(state_df, val_weeks=16)
    logger.info(f"[{state}] Train: {len(train_df)} weeks, Val: {len(val_df)} weeks")

    predictions = {}   # val predictions
    future_preds = {}  # 8-week future forecasts
    model_results = []
    trained_models = {}

    # --- SARIMA ---
    try:
        from src.models.arima_model import fit_sarima, forecast_sarima, save_sarima
        sarima_dict = fit_sarima(train_df.set_index("date")["sales"], state)
        if sarima_dict:
            # validate
            val_preds_sarima = forecast_sarima(sarima_dict, n_steps=len(val_df))
            actual = val_df["sales"].values
            result = evaluate_model(actual[:len(val_preds_sarima)], val_preds_sarima, "sarima")
            model_results.append(result)
            predictions["sarima"] = val_preds_sarima

            # future forecast
            # refit on full data for future prediction
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            full_series = state_df.set_index("date")["sales"]
            full_model = SARIMAX(
                full_series,
                order=sarima_dict["order"],
                seasonal_order=sarima_dict["seasonal_order"],
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)
            sarima_dict_full = {"model": full_model, **{k: v for k, v in sarima_dict.items() if k != "model"}}
            future_preds["sarima"] = forecast_sarima(sarima_dict_full, n_steps=8)

            save_sarima(sarima_dict_full, state, models_dir)
            trained_models["sarima"] = sarima_dict_full
    except Exception as e:
        logger.error(f"[{state}] SARIMA error: {e}")

    # --- Prophet ---
    try:
        from src.models.prophet_model import fit_prophet, forecast_prophet, save_prophet
        prophet_dict = fit_prophet(train_df, state)
        if prophet_dict:
            last_train_date = train_df["date"].max()
            val_preds_prophet = forecast_prophet(prophet_dict, last_train_date, n_steps=len(val_df))
            actual = val_df["sales"].values
            result = evaluate_model(actual[:len(val_preds_prophet)], val_preds_prophet, "prophet")
            model_results.append(result)
            predictions["prophet"] = val_preds_prophet

            # future — refit on full data
            prophet_full = fit_prophet(state_df, state)
            last_date = state_df["date"].max()
            future_preds["prophet"] = forecast_prophet(prophet_full, last_date, n_steps=8)

            save_prophet(prophet_full, state, models_dir)
            trained_models["prophet"] = prophet_full
    except Exception as e:
        logger.error(f"[{state}] Prophet error: {e}")

    # --- XGBoost ---
    try:
        from src.models.xgboost_model import fit_xgboost, forecast_xgboost, save_xgboost, get_feature_importance
        xgb_dict = fit_xgboost(train_df, state)
        if xgb_dict:
            val_preds_xgb = forecast_xgboost(xgb_dict, n_steps=len(val_df))
            actual = val_df["sales"].values
            result = evaluate_model(actual[:len(val_preds_xgb)], val_preds_xgb, "xgboost")
            model_results.append(result)
            predictions["xgboost"] = val_preds_xgb

            # feature importance plot
            fi_df = get_feature_importance(xgb_dict)
            plots_dir = os.path.join(outputs_dir, "plots")
            plot_feature_importance(fi_df, state, plots_dir)

            # future — refit on full data
            xgb_full = fit_xgboost(state_df, state)
            future_preds["xgboost"] = forecast_xgboost(xgb_full, n_steps=8)

            save_xgboost(xgb_full, state, models_dir)
            trained_models["xgboost"] = xgb_full
    except Exception as e:
        logger.error(f"[{state}] XGBoost error: {e}")

    # --- LSTM ---
    if not skip_lstm:
        try:
            from src.models.lstm_model import fit_lstm, forecast_lstm, save_lstm
            lstm_dict = fit_lstm(train_df, state)
            if lstm_dict:
                val_preds_lstm = forecast_lstm(lstm_dict, n_steps=len(val_df))
                actual = val_df["sales"].values
                result = evaluate_model(actual[:len(val_preds_lstm)], val_preds_lstm, "lstm")
                model_results.append(result)
                predictions["lstm"] = val_preds_lstm

                # future — refit on full data
                lstm_full = fit_lstm(state_df, state)
                future_preds["lstm"] = forecast_lstm(lstm_full, n_steps=8)

                save_lstm(lstm_full, state, models_dir)
                trained_models["lstm"] = lstm_full
        except Exception as e:
            logger.error(f"[{state}] LSTM error: {e}")
    else:
        logger.info(f"[{state}] Skipping LSTM (skip_lstm=True)")

    if not model_results:
        logger.error(f"[{state}] All models failed!")
        return None

    # --- Evaluation and model selection ---
    comparison_df = build_comparison_table(model_results)
    best_model = select_best_model(comparison_df)
    log_state_results(state, comparison_df, best_model)

    # --- Save which model is best for this state ---
    best_info = {
        "state": state,
        "best_model": best_model,
        "metrics": comparison_df.to_dict(orient="records"),
        "future_forecast_8w": future_preds.get(best_model, [None]).__class__ == np.ndarray
            and future_preds.get(best_model, []).tolist()
            or [],
    }

    # save best model info
    os.makedirs(models_dir, exist_ok=True)
    info_path = os.path.join(models_dir, f"best_model_{state.replace(' ', '_')}.json")
    with open(info_path, "w") as f:
        json.dump(best_info, f, indent=2)

    # --- Visualizations ---
    try:
        plots_dir = os.path.join(outputs_dir, "plots")
        plot_forecast_vs_actual(
            state=state,
            train_df=train_df,
            val_df=val_df,
            predictions=predictions,
            future_preds=future_preds,
            save_dir=plots_dir,
            best_model=best_model,
        )
    except Exception as e:
        logger.warning(f"[{state}] Plot failed: {e}")

    return {
        "state": state,
        "best_model": best_model,
        "comparison": comparison_df,
        "future_preds": future_preds,
        "trained_models": trained_models,
    }


def run_full_pipeline(
    data_path: str,
    states_to_train: Optional[list] = None,
    skip_lstm: bool = False,
):
    """
    Run training for all states (or a subset).
    Saves a global comparison CSV at the end.
    """
    logger.info("Starting full training pipeline...")

    state_series = preprocess_all_states(data_path)

    if states_to_train:
        state_series = {s: df for s, df in state_series.items() if s in states_to_train}
        logger.info(f"Training subset: {list(state_series.keys())}")

    all_results = []
    state_best_models = {}

    for state, state_df in state_series.items():
        try:
            result = train_one_state(
                state=state,
                state_df=state_df,
                skip_lstm=skip_lstm,
            )
            if result is None:
                continue

            state_best_models[state] = result["best_model"]
            for row in result["comparison"].to_dict(orient="records"):
                row["state"] = state
                all_results.append(row)
        except Exception as e:
            logger.error(f"Pipeline failed for {state}: {e}", exc_info=True)

    if not all_results:
        logger.error("No results to save!")
        return

    # --- Global comparison table ---
    all_df = pd.DataFrame(all_results)
    metrics_dir = os.path.join(OUTPUTS_DIR, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    all_df.to_csv(os.path.join(metrics_dir, "all_metrics.csv"), index=False)
    logger.info(f"Saved all metrics to outputs/metrics/all_metrics.csv")

    # model comparison chart
    plots_dir = os.path.join(OUTPUTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    try:
        plot_model_comparison(all_df, plots_dir)
    except Exception as e:
        logger.warning(f"Model comparison plot failed: {e}")

    # save overall best model mapping
    mapping_path = os.path.join(MODELS_DIR, "state_best_models.json")
    with open(mapping_path, "w") as f:
        json.dump(state_best_models, f, indent=2)
    logger.info(f"State → best model mapping saved to {mapping_path}")

    # print summary
    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"  States trained: {len(state_best_models)}")
    print(f"\n  Best model counts:")
    from collections import Counter
    for model, count in Counter(state_best_models.values()).items():
        print(f"    {model}: {count} states")
    print("="*60 + "\n")

    return state_best_models
