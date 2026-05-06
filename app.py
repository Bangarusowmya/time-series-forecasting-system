"""
FastAPI app for serving time series forecasts.

Endpoints:
  POST /forecast         - Get 8-week forecast for a given state
  GET  /states           - List all available states
  GET  /health           - Health check
  GET  /model-info/{state} - Get best model and metrics for a state

Run with:
  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json

# make sure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from src.predict_pipeline import predict_state, get_all_available_states
from src.logger import get_logger

logger = get_logger("api")

app = FastAPI(
    title="Sales Forecasting API",
    description="8-week sales forecasting for US states using SARIMA, Prophet, XGBoost, and LSTM",
    version="1.0.0",
)


# --- Request / Response models ---

class ForecastRequest(BaseModel):
    state: str
    n_weeks: Optional[int] = 8

    class Config:
        json_schema_extra = {
            "example": {
                "state": "Texas",
                "n_weeks": 8,
            }
        }


class ForecastItem(BaseModel):
    week: int
    date: str
    forecast_sales: float


class ForecastResponse(BaseModel):
    state: str
    model_used: str
    forecast_start: str
    forecast_end: str
    forecast: list


# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "sales-forecasting-api"}


@app.get("/states")
def list_states():
    """Returns all states that have been trained and are ready for forecasting."""
    states = get_all_available_states()
    if not states:
        return JSONResponse(
            status_code=503,
            content={
                "error": "No trained models found. Run 'python main.py' first.",
                "states": [],
            },
        )
    return {"count": len(states), "states": states}


@app.post("/forecast")
def get_forecast(request: ForecastRequest):
    """
    Generate n_weeks sales forecast for a given state.
    
    The best model (SARIMA/Prophet/XGBoost/LSTM) is automatically selected
    based on validation performance during training.
    """
    state = request.state.strip()
    n_weeks = request.n_weeks or 8

    if n_weeks < 1 or n_weeks > 52:
        raise HTTPException(
            status_code=400,
            detail="n_weeks must be between 1 and 52"
        )

    logger.info(f"Forecast request: state={state}, n_weeks={n_weeks}")

    try:
        result = predict_state(state, n_weeks=n_weeks)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Forecast failed for {state}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")


@app.get("/model-info/{state}")
def model_info(state: str):
    """Returns the best model and evaluation metrics for a given state."""
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    info_path = os.path.join(models_dir, f"best_model_{state.replace(' ', '_')}.json")

    if not os.path.exists(info_path):
        raise HTTPException(
            status_code=404,
            detail=f"No model info found for '{state}'. Run training first."
        )

    with open(info_path) as f:
        info = json.load(f)
    return info


@app.get("/")
def root():
    return {
        "message": "Sales Forecasting API",
        "docs": "/docs",
        "endpoints": ["/forecast", "/states", "/model-info/{state}", "/health"],
    }
