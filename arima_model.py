"""
SARIMA model trainer.

Uses auto_arima from pmdarima if available, otherwise falls back to a 
reasonable default order. Weekly data with annual seasonality (m=52) can be 
slow to fit so we limit the search space a bit.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger("arima_model")

# default order — works decently for most retail/sales series
DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL_ORDER = (1, 1, 0, 52)


def fit_sarima(train_series: pd.Series, state: str) -> dict:
    """
    Fit SARIMA model on training data.
    Returns a dict with model and fit info.
    """
    logger.info(f"[{state}] Fitting SARIMA...")

    # Try auto order selection first, fall back to defaults if it errors out
    try:
        import pmdarima as pm
        auto_model = pm.auto_arima(
            train_series,
            seasonal=True,
            m=52,
            d=1,
            D=1,
            start_p=0, max_p=2,
            start_q=0, max_q=2,
            start_P=0, max_P=1,
            start_Q=0, max_Q=1,
            stepwise=True,
            information_criterion="aic",
            suppress_warnings=True,
            error_action="ignore",
            n_jobs=1,
        )
        order = auto_model.order
        seasonal_order = auto_model.seasonal_order
        logger.info(f"[{state}] auto_arima selected order={order}, seasonal={seasonal_order}")
    except ImportError:
        order = DEFAULT_ORDER
        seasonal_order = DEFAULT_SEASONAL_ORDER
        logger.info(f"[{state}] pmdarima not installed, using defaults")
    except Exception as e:
        logger.warning(f"[{state}] auto_arima failed ({e}), using defaults")
        order = DEFAULT_ORDER
        seasonal_order = DEFAULT_SEASONAL_ORDER

    try:
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False, maxiter=200)
        logger.info(f"[{state}] SARIMA fit done. AIC: {result.aic:.2f}")
        return {"model": result, "order": order, "seasonal_order": seasonal_order}
    except Exception as e:
        logger.error(f"[{state}] SARIMA fitting failed: {e}")
        return None


def forecast_sarima(model_dict: dict, n_steps: int = 8) -> np.ndarray:
    """Forecast n_steps ahead. Returns raw numpy array of predictions."""
    result = model_dict["model"]
    forecast = result.get_forecast(steps=n_steps)
    preds = forecast.predicted_mean.values
    # clip negatives — sales can't be negative
    preds = np.clip(preds, 0, None)
    return preds


def save_sarima(model_dict: dict, state: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"sarima_{state.replace(' ', '_')}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model_dict, f)
    logger.info(f"[{state}] SARIMA model saved to {path}")
    return path


def load_sarima(state: str, save_dir: str) -> dict:
    path = os.path.join(save_dir, f"sarima_{state.replace(' ', '_')}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
