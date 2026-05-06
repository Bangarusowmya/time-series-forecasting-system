"""
Facebook Prophet model.

Prophet handles trend + seasonality well out of the box and is pretty
forgiving with irregularly spaced data. Weekly data with US holidays fits
nicely into Prophet's framework.
"""

import os
import pickle
import pandas as pd
import numpy as np
from prophet import Prophet
from src.logger import get_logger

logger = get_logger("prophet_model")


def fit_prophet(train_df: pd.DataFrame, state: str) -> dict:
    """
    Train Prophet model.

    train_df should contain:
    - date
    - sales
    """

    logger.info(f"[{state}] Fitting Prophet...")

    # Prophet requires columns: ds and y
    prophet_df = train_df[["date", "sales"]].rename(
        columns={
            "date": "ds",
            "sales": "y"
        }
    )

    # Create Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        stan_backend="CMDSTANPY"
    )

    # Add US holidays
    try:
        model.add_country_holidays(country_name="US")
    except Exception:
        pass

    # Train model
    model.fit(prophet_df, show_progress=False)

    logger.info(f"[{state}] Prophet fit complete")

    return {"model": model}


def forecast_prophet(
    model_dict: dict,
    last_date: pd.Timestamp,
    n_steps: int = 8
) -> np.ndarray:
    """
    Generate future forecasts.
    """

    model = model_dict["model"]

    # Future weekly dates
    future_dates = pd.date_range(
        start=last_date,
        periods=n_steps + 1,
        freq="W"
    )[1:]

    future_df = pd.DataFrame({
        "ds": future_dates
    })

    forecast = model.predict(future_df)

    preds = forecast["yhat"].values

    # Remove negative predictions
    preds = np.clip(preds, 0, None)

    return preds


def save_prophet(
    model_dict: dict,
    state: str,
    save_dir: str
):
    """
    Save trained Prophet model.
    """

    os.makedirs(save_dir, exist_ok=True)

    path = os.path.join(
        save_dir,
        f"prophet_{state.replace(' ', '_')}.pkl"
    )

    with open(path, "wb") as f:
        pickle.dump(model_dict, f)

    logger.info(f"[{state}] Prophet saved to {path}")

    return path


def load_prophet(
    state: str,
    save_dir: str
) -> dict:
    """
    Load saved Prophet model.
    """

    path = os.path.join(
        save_dir,
        f"prophet_{state.replace(' ', '_')}.pkl"
    )

    with open(path, "rb") as f:
        return pickle.load(f)