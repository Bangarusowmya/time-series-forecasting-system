"""
Prediction pipeline for serving forecasts.

Loads trained models and generates 8-week ahead forecasts for a given state.
Used by the FastAPI app.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.data_preprocessing import preprocess_all_states
from src.logger import get_logger

logger = get_logger("predict_pipeline")

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sales_data.xlsx")


def get_best_model_name(state: str) -> str:
    """Load the saved best model name for a state."""
    info_path = os.path.join(MODELS_DIR, f"best_model_{state.replace(' ', '_')}.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"No trained model found for '{state}'. "
            "Run main.py first to train models."
        )
    with open(info_path) as f:
        info = json.load(f)
    return info["best_model"]


def load_state_data(state: str) -> pd.DataFrame:
    """Load preprocessed data for a specific state."""
    state_series = preprocess_all_states(DATA_PATH)
    if state not in state_series:
        available = sorted(state_series.keys())
        raise ValueError(
            f"State '{state}' not found. Available states:\n{available}"
        )
    return state_series[state]


def predict_state(state: str, n_weeks: int = 8) -> dict:
    """
    Main prediction function. Returns forecast for n_weeks ahead.
    Loads the best model for the state and runs inference.
    """
    logger.info(f"[{state}] Generating {n_weeks}-week forecast...")

    best_model = get_best_model_name(state)
    logger.info(f"[{state}] Using model: {best_model}")

    state_df = load_state_data(state)
    last_date = state_df["date"].max()
    forecast_dates = pd.date_range(start=last_date, periods=n_weeks + 1, freq="W")[1:]

    predictions = None

    if best_model == "sarima":
        from src.models.arima_model import load_sarima, forecast_sarima
        model_dict = load_sarima(state, MODELS_DIR)
        predictions = forecast_sarima(model_dict, n_steps=n_weeks)

    elif best_model == "prophet":
        from src.models.prophet_model import load_prophet, forecast_prophet
        model_dict = load_prophet(state, MODELS_DIR)
        predictions = forecast_prophet(model_dict, last_date, n_steps=n_weeks)

    elif best_model == "xgboost":
        from src.models.xgboost_model import load_xgboost, forecast_xgboost
        model_dict = load_xgboost(state, MODELS_DIR)
        predictions = forecast_xgboost(model_dict, n_steps=n_weeks)

    elif best_model == "lstm":
        from src.models.lstm_model import load_lstm, forecast_lstm
        model_dict = load_lstm(state, MODELS_DIR)
        predictions = forecast_lstm(model_dict, n_steps=n_weeks)

    else:
        raise ValueError(f"Unknown model type: {best_model}")

    if predictions is None or len(predictions) == 0:
        raise RuntimeError(f"Model {best_model} returned empty predictions for {state}")

    forecast_output = [
        {
            "week": i + 1,
            "date": str(forecast_dates[i].date()),
            "forecast_sales": round(float(p), 2),
        }
        for i, p in enumerate(predictions[:n_weeks])
    ]

    return {
        "state": state,
        "model_used": best_model,
        "forecast_start": str(forecast_dates[0].date()),
        "forecast_end": str(forecast_dates[n_weeks - 1].date()),
        "forecast": forecast_output,
    }


def get_all_available_states() -> list:
    """Return list of states that have been trained."""
    if not os.path.exists(MODELS_DIR):
        return []
    mapping_path = os.path.join(MODELS_DIR, "state_best_models.json")
    if not os.path.exists(mapping_path):
        return []
    with open(mapping_path) as f:
        mapping = json.load(f)
    return sorted(mapping.keys())
