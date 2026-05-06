"""
LSTM model for sales forecasting.

Uses a simple 2-layer LSTM. For 8-week ahead forecasting on ~100-150 weekly
data points, we don't need a complex architecture — keeping it simple avoids
overfitting and trains faster.
"""

import os
import pickle
import numpy as np
import pandas as pd
from src.feature_engineering import prepare_lstm_sequences, time_series_split
from src.logger import get_logger

logger = get_logger("lstm_model")

LOOKBACK = 12   # use 12 weeks of history to predict next week
EPOCHS = 80
BATCH_SIZE = 16


def build_lstm_model(lookback: int = LOOKBACK):
    """Build and compile the LSTM model."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    tf.random.set_seed(42)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.1),
        Dense(16, activation="relu"),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def fit_lstm(train_df: pd.DataFrame, state: str) -> dict:
    """
    Prepare sequences and train the LSTM.
    Uses 10% of training data for validation to monitor early stopping.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    except ImportError:
        raise ImportError("tensorflow not installed. Run: pip install tensorflow")

    logger.info(f"[{state}] Training LSTM (lookback={LOOKBACK}, epochs={EPOCHS})...")

    sales = train_df["sales"].values

    if len(sales) < LOOKBACK + 10:
        logger.error(f"[{state}] Not enough data for LSTM (need >{LOOKBACK + 10}, got {len(sales)})")
        return None

    X, y, scaler = prepare_lstm_sequences(sales, lookback=LOOKBACK)

    # split into train/val (no shuffle!)
    val_size = max(8, int(0.1 * len(X)))
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

    model = build_lstm_model(LOOKBACK)

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-5, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=0,
    )

    best_val_loss = min(history.history["val_loss"])
    logger.info(f"[{state}] LSTM training done. Best val_loss: {best_val_loss:.6f}")

    return {
        "model": model,
        "scaler": scaler,
        "lookback": LOOKBACK,
        "last_sequence": sales[-LOOKBACK:],  # for recursive forecasting
    }


def forecast_lstm(model_dict: dict, n_steps: int = 8) -> np.ndarray:
    """
    Recursive forecast: predict 1 step, append, slide window, repeat.
    """
    model = model_dict["model"]
    scaler = model_dict["scaler"]
    lookback = model_dict["lookback"]
    last_seq = model_dict["last_sequence"].copy()

    # scale the input sequence
    scaled_seq = scaler.transform(last_seq.reshape(-1, 1)).flatten()

    preds_scaled = []
    current_window = list(scaled_seq)

    for _ in range(n_steps):
        x_input = np.array(current_window[-lookback:]).reshape(1, lookback, 1)
        pred_scaled = model.predict(x_input, verbose=0)[0][0]
        preds_scaled.append(pred_scaled)
        current_window.append(pred_scaled)

    # inverse transform
    preds = scaler.inverse_transform(
        np.array(preds_scaled).reshape(-1, 1)
    ).flatten()
    preds = np.clip(preds, 0, None)
    return preds


def save_lstm(model_dict: dict, state: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    state_clean = state.replace(" ", "_")

    # save keras model separately (can't pickle TF models reliably)
    model_path = os.path.join(save_dir, f"lstm_{state_clean}.h5")
    model_dict["model"].save(model_path)

    # save everything else
    meta_path = os.path.join(save_dir, f"lstm_{state_clean}_meta.pkl")
    meta = {
        "scaler": model_dict["scaler"],
        "lookback": model_dict["lookback"],
        "last_sequence": model_dict["last_sequence"],
        "model_path": model_path,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    logger.info(f"[{state}] LSTM saved to {model_path}")
    return model_path


def load_lstm(state: str, save_dir: str) -> dict:
    import tensorflow as tf
    state_clean = state.replace(" ", "_")

    meta_path = os.path.join(save_dir, f"lstm_{state_clean}_meta.pkl")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    model = tf.keras.models.load_model(meta["model_path"])
    return {
        "model": model,
        "scaler": meta["scaler"],
        "lookback": meta["lookback"],
        "last_sequence": meta["last_sequence"],
    }
