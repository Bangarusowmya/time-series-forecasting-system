"""
Feature engineering for the ML-based models (XGBoost, LSTM).

ARIMA/SARIMA and Prophet don't use these — they work directly on the time series.
For XGBoost we need tabular features. For LSTM we need scaled sequences.

Important: all features are built ONLY on training data to avoid leakage.
The same transformations are applied to validation/test data using stats from train.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.logger import get_logger

logger = get_logger("feature_engineering")

# US federal holidays (month, day) tuples — covers the major ones
# Using a simple lookup instead of the holidays library to avoid dependency issues
_US_FIXED_HOLIDAYS = {
    (1, 1),   # New Year's Day
    (7, 4),   # Independence Day
    (11, 11), # Veterans Day
    (12, 25), # Christmas
    (12, 24), # Christmas Eve (observed)
    (12, 31), # New Year's Eve
}

def _is_near_us_holiday(date: pd.Timestamp) -> bool:
    """Check if a date is within 3 days of a major US holiday."""
    for delta in range(-3, 4):
        d = date + pd.Timedelta(days=delta)
        if (d.month, d.day) in _US_FIXED_HOLIDAYS:
            return True
        # Thanksgiving: 4th Thursday of November
        if d.month == 11 and d.dayofweek == 3:
            # check if it's the 4th Thursday
            if 22 <= d.day <= 28:
                return True
        # Labor Day: 1st Monday of September
        if d.month == 9 and d.dayofweek == 0 and 1 <= d.day <= 7:
            return True
        # Memorial Day: last Monday of May
        if d.month == 5 and d.dayofweek == 0 and d.day >= 25:
            return True
    return False


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Month and day-of-week as cyclical features (sin/cos) + raw integers."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter

    # cyclical encoding for month (helps tree models and LSTM handle wrap-around)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df


def add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Simple binary flag: 1 if the week contains or is near a US holiday."""
    df = df.copy()
    df["is_holiday"] = df["date"].apply(
        lambda d: 1 if (d.month, d.day) in _US_FIXED_HOLIDAYS else 0
    )
    df["near_holiday"] = df["date"].apply(
        lambda d: 1 if _is_near_us_holiday(d) else 0
    )
    return df


def add_lag_features(df: pd.DataFrame, lags: list = None) -> pd.DataFrame:
    """
    Lag features — past sales values as predictors.
    Default: lag_1 (1 week), lag_4 (~1 month), lag_8 (~2 months), lag_52 (~1 year)
    
    The original spec asks for lag_1, lag_7, lag_30 but since we're working
    with WEEKLY data, I'm mapping those to weekly-equivalent lags:
        lag_1  → 1 week back
        lag_4  → ~1 month back  (was lag_7 days, adapted to weekly)
        lag_13 → ~1 quarter back
        lag_52 → ~1 year back
    """
    if lags is None:
        lags = [1, 4, 8, 13, 52]  # weeks

    df = df.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling mean and std — helps capture local trend and volatility."""
    df = df.copy()

    # 4-week rolling (1 month)
    df["rolling_mean_4"] = df["sales"].shift(1).rolling(window=4).mean()
    df["rolling_std_4"] = df["sales"].shift(1).rolling(window=4).std()

    # 12-week rolling (quarter)
    df["rolling_mean_12"] = df["sales"].shift(1).rolling(window=12).mean()
    df["rolling_std_12"] = df["sales"].shift(1).rolling(window=12).std()

    # year-over-year ratio (if we have 52 weeks of history)
    df["yoy_ratio"] = df["sales"].shift(52) / (df["sales"].shift(53) + 1e-6)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature pipeline. Input df must have 'date' and 'sales' columns.
    Returns df with all features added. NaN rows from lags are dropped.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df = add_time_features(df)
    df = add_holiday_flag(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # drop rows where lag features are NaN (first ~52 rows)
    before = len(df)
    df = df.dropna()
    after = len(df)
    logger.debug(f"Dropped {before - after} rows with NaN features")

    return df


def get_feature_columns() -> list:
    """Returns the list of feature column names used in model training."""
    lag_cols = [f"lag_{l}" for l in [1, 4, 8, 13, 52]]
    rolling_cols = [
        "rolling_mean_4", "rolling_std_4",
        "rolling_mean_12", "rolling_std_12",
        "yoy_ratio"
    ]
    time_cols = [
        "month", "day_of_week", "week_of_year", "quarter",
        "month_sin", "month_cos", "dow_sin", "dow_cos"
    ]
    holiday_cols = ["is_holiday", "near_holiday"]
    return lag_cols + rolling_cols + time_cols + holiday_cols


def time_series_split(df: pd.DataFrame, val_weeks: int = 16):
    """
    Simple time-based train/val split. No shuffling, no data leakage.
    val_weeks: how many weeks to hold out for validation
    """
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = len(df) - val_weeks
    train = df.iloc[:split_idx].copy()
    val = df.iloc[split_idx:].copy()
    return train, val


def prepare_lstm_sequences(series: np.ndarray, lookback: int = 12):
    """
    Reshape sales data into (samples, timesteps, features) for LSTM.
    Returns X, y and the scaler (need to save scaler for inverse transform).
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y, scaler
