"""Model API: loads the registered model from MLflow and serves predictions."""
import os
from typing import Any

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# These come from docker-compose / .env. MODEL_NAME must match the name
# Anto registers in MLflow (must use the same as him).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "insy684-model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "champion")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
app = FastAPI(title="Model API")
Instrumentator().instrument(app).expose(app)
_model = None


def get_model():
    """Load the model the first time it's needed, not at startup,
    so the API still starts even if no model is registered yet."""
    global _model
    if _model is None:
        uri = f"models:/{MODEL_NAME}@{MODEL_VERSION}"
        try:
            _model = mlflow.pyfunc.load_model(uri)
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Model not available at {uri}: {exc}"
            )
    return _model


class PredictRequest(BaseModel):
    # One row of input features, e.g. {"age": 34, "income": 52000}.
    # Column names must match what the model was trained on (ask Jose/Anto).
    features: dict[str, Any]


class PredictResponse(BaseModel):
    prediction: float


@app.get("/health")
def health():
    loaded = _model is not None
    return {"status": "ok", "model_name": MODEL_NAME, "model_loaded": loaded}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = get_model()
    df = pd.DataFrame([req.features]).astype(float)
    try:
        result = model.predict(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Bad input: {exc}")
    return PredictResponse(prediction=float(result[0]))
