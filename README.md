# 📊 Time Series Forecasting System

An end-to-end sales forecasting system that predicts **8 weeks of future sales for each US state** using historical data. Built with SARIMA, Facebook Prophet, XGBoost, and LSTM — with automated model selection and a REST API for serving predictions.

---

## Project Overview

The dataset contains weekly beverage sales across 43 US states from 2019 to 2023. This system:

- Cleans and preprocesses the raw data (handles irregular dates, gaps, outliers)
- Engineers time-based features: lag values, rolling stats, holidays, seasonality
- Trains 4 different forecasting models on each state
- Evaluates all models on a held-out validation set using RMSE, MAE, MAPE
- Automatically picks the best-performing model per state
- Saves trained models to disk
- Exposes predictions via a FastAPI REST endpoint

---

## Architecture

```
Raw Excel Data
      │
      ▼
Data Preprocessing (fill gaps, resample weekly, clip negatives)
      │
      ▼
Feature Engineering (lags, rolling stats, time features, holidays)
      │
      ├──► SARIMA/SARIMAX
      ├──► Facebook Prophet  
      ├──► XGBoost (lag-based regression)
      └──► LSTM (sequence model)
              │
              ▼
        Evaluation (RMSE, MAE, MAPE on val set)
              │
              ▼
        Best Model Selection (per state)
              │
              ▼
        FastAPI (POST /forecast)
```

---

## Project Structure

```
time-series-forecasting-system/
│
├── data/
│   └── sales_data.xlsx          # Raw input data
│
├── notebooks/
│   └── EDA.ipynb                # Exploratory analysis
│
├── src/
│   ├── data_preprocessing.py    # Load, clean, resample data
│   ├── feature_engineering.py   # Lag features, rolling stats, holidays
│   ├── evaluation.py            # RMSE, MAE, MAPE + best model selection
│   ├── train_pipeline.py        # Orchestrates training for all states
│   ├── predict_pipeline.py      # Load models, generate forecasts
│   ├── visualization.py         # Forecast plots, model comparison
│   ├── logger.py                # Logging setup
│   └── models/
│       ├── arima_model.py       # SARIMA training + forecasting
│       ├── prophet_model.py     # Prophet training + forecasting
│       ├── xgboost_model.py     # XGBoost with lag features
│       └── lstm_model.py        # LSTM with TensorFlow/Keras
│
├── api/
│   └── app.py                   # FastAPI application
│
├── models/                      # Saved trained models (auto-created)
│
├── outputs/
│   ├── plots/                   # Forecast plots, feature importance
│   └── metrics/                 # all_metrics.csv
│
├── logs/                        # Daily log files
│
├── main.py                      # Entry point for training
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/Bangaeusowmya/time-series-forecasting-system.git
cd time-series-forecasting-system
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow installation depends on your OS/hardware. If you're on Apple Silicon:
> ```bash
> pip install tensorflow-macos tensorflow-metal
> ```

### 4. (Optional) Install pmdarima for auto ARIMA order selection

```bash
pip install pmdarima
```

---

## Execution Steps

### Step 1 — Train Models

Train all states (includes LSTM, takes ~30–60 min depending on hardware):

```bash
python main.py
```

Quick test with 3 states and no LSTM (~2–3 min):

```bash
python main.py --states "Texas" "California" "Florida" --skip-lstm
```

Train specific states only:

```bash
python main.py --states "Texas" "New York" "Illinois"
```

### Step 2 — Start the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

---

## API Usage

### POST /forecast

Get 8-week sales forecast for a state.

**Request:**
```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{"state": "Texas"}'
```

**Response:**
```json
{
  "state": "Texas",
  "model_used": "prophet",
  "forecast_start": "2024-01-07",
  "forecast_end": "2024-02-25",
  "forecast": [
    {"week": 1, "date": "2024-01-07", "forecast_sales": 485234567.0},
    {"week": 2, "date": "2024-01-14", "forecast_sales": 491023456.0},
    ...
  ]
}
```

### GET /states

List all trained states:

```bash
curl http://localhost:8000/states
```

### GET /model-info/{state}

Get model details and metrics for a state:

```bash
curl http://localhost:8000/model-info/Texas
```

### GET /health

```bash
curl http://localhost:8000/health
```

---

## Models

| Model | Type | Notes |
|-------|------|-------|
| SARIMA | Statistical | Handles trend + seasonality via differencing and MA/AR terms |
| Prophet | Statistical/ML | Additive decomposition with US holidays baked in |
| XGBoost | ML (tabular) | Lag features + rolling stats as input. Recursive multi-step forecast |
| LSTM | Deep Learning | Sequence-to-one with 12-week lookback window |

### Feature Engineering (for XGBoost/LSTM)

| Feature | Description |
|---------|-------------|
| lag_1 | Sales 1 week ago |
| lag_4 | Sales ~1 month ago |
| lag_8 | Sales ~2 months ago |
| lag_13 | Sales ~1 quarter ago |
| lag_52 | Sales ~1 year ago |
| rolling_mean_4 | 4-week rolling average |
| rolling_std_4 | 4-week rolling standard deviation |
| rolling_mean_12 | 12-week rolling average |
| rolling_std_12 | 12-week rolling standard deviation |
| month / month_sin / month_cos | Month-based seasonality (cyclical encoding) |
| day_of_week | Day of week |
| week_of_year | Week number |
| is_holiday | US federal holiday flag |
| near_holiday | Within 3 days of a US holiday |
| yoy_ratio | Year-over-year sales ratio |

---

## Outputs

After training, the following files are generated:

```
models/
  sarima_Texas.pkl
  prophet_Texas.pkl
  xgboost_Texas.pkl
  lstm_Texas.h5
  lstm_Texas_meta.pkl
  best_model_Texas.json
  state_best_models.json
  ... (one set per state)

outputs/
  plots/
    forecast_Texas.png          # Val predictions + 8-week future forecast
    feature_importance_Texas.png
    model_comparison.png        # Bar chart of avg metrics across states
  metrics/
    all_metrics.csv             # Full comparison table
```

---

## Screenshots

> 

**Forecast vs Actual:**
`outputs/plots/forecast_Texas.png`

**Model Comparison:**
`outputs/plots/model_comparison.png`

**XGBoost Feature Importance:**
`outputs/plots/feature_importance_Texas.png`

---

## Future Improvements

- Add a Streamlit dashboard for visual forecasts
- Support category-level forecasting (currently aggregated to state-total)
- Add confidence intervals to API response
- Implement hyperparameter tuning with Optuna
- Add support for external regressors (price, promotions, weather)
- Set up model retraining triggers when new data arrives
- Dockerize the API for easier deployment

---

## Tech Stack

- **Python 3.10+**
- pandas, numpy, scikit-learn
- statsmodels (SARIMA)
- prophet (Facebook Prophet)
- xgboost
- tensorflow/keras (LSTM)
- fastapi + uvicorn (REST API)
- matplotlib + seaborn (visualizations)
- holidays (US federal holiday detection)
- joblib, pickle (model serialization)

---

## Author

Built as a portfolio project demonstrating end-to-end ML engineering for time series forecasting.

