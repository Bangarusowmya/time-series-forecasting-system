"""
Data preprocessing pipeline.

Loads the raw Excel file, handles missing dates, fills gaps,
and returns a clean per-state weekly time series ready for feature engineering.
"""

import os
import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger("preprocessing")


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Read the Excel file and do basic type fixes."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip().str.lower()

    # rename to standard names
    rename_map = {
        "state": "state",
        "date": "date",
        "total": "sales",
        "category": "category",
    }
    df = df.rename(columns=rename_map)

    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")

    logger.info(f"Loaded {len(df)} rows, {df['state'].nunique()} states")
    return df


def fill_missing_dates(state_df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    For a single state's data, resample to weekly frequency.
    Gaps in dates get filled with linear interpolation — simple but good enough
    for sales data that's relatively smooth.
    """
    state_df = state_df.sort_values("date").set_index("date")
    # resample to weekly Sunday frequency
    state_df = state_df["sales"].resample(freq).mean()
    # interpolate small gaps, forward fill anything at the edges
    state_df = state_df.interpolate(method="linear", limit=4)
    state_df = state_df.ffill().bfill()
    return state_df.reset_index().rename(columns={"date": "date", "sales": "sales"})


def preprocess_all_states(filepath: str) -> dict:
    """
    Main preprocessing entry point. Returns a dict like:
        { "Alabama": pd.DataFrame, "Texas": pd.DataFrame, ... }
    Each DataFrame has columns: date, sales
    """
    raw = load_raw_data(filepath)

    # aggregate across categories (only 'Beverages' here but keeping it flexible)
    agg = raw.groupby(["state", "date"])["sales"].sum().reset_index()

    state_series = {}
    skipped = []

    for state, grp in agg.groupby("state"):
        grp = grp[["date", "sales"]].drop_duplicates("date")

        if len(grp) < 30:
            # not enough data to train anything reasonable
            logger.warning(f"Skipping {state} — only {len(grp)} data points")
            skipped.append(state)
            continue

        cleaned = fill_missing_dates(grp)
        # drop any remaining NaNs (shouldn't be many after interpolation)
        cleaned = cleaned.dropna(subset=["sales"])

        # sanity check: negative sales don't make sense
        neg_count = (cleaned["sales"] < 0).sum()
        if neg_count > 0:
            logger.warning(f"{state}: found {neg_count} negative sales, clipping to 0")
            cleaned["sales"] = cleaned["sales"].clip(lower=0)

        state_series[state] = cleaned
        logger.info(f"{state}: {len(cleaned)} weekly records after preprocessing")

    if skipped:
        logger.warning(f"Skipped states due to insufficient data: {skipped}")

    return state_series


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "sales_data.xlsx")
    series = preprocess_all_states(data_path)
    print(f"\nPreprocessed {len(series)} states")
    sample_state = list(series.keys())[0]
    print(f"\nSample ({sample_state}):")
    print(series[sample_state].tail())
