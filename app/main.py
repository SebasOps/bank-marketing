"""
API de inferencia - modelo de clasificación de Bank-Marketing servido con MLflow + FastAPI.

Endpoint: 
    GET  /health   -> verifica que el servicio y el modelo están cargados
    POST /predict  -> recibe features, devuelve predicción + probabilidad
"""

# Imports
import sys
import json
import time
from pathlib import Path
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from datetime import datetime, timezone


# Calcular ruta raíz y agregarla a sys.path para poder importar desde src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
sys.path.append(str(PROJECT_ROOT)) 

from src.quality.clean import lower_case


# Ruta del modelo
MODEL_DIR = PROJECT_ROOT / "model_artifact"


# Información de la API
app = FastAPI(
    title="API de inferencia - Bank Marketing",
    description="Sirve un modelo scikit-learn entrenado y registrado en MLflow.",
    version="1.0.0",
)


# Middleware 
# Intercepta cada request, mide latencia 
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
REQUEST_LOG_PATH = LOG_DIR / "requests.jsonl"

@app.middleware("http")
async def log_requests(request, call_next):
    start = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": request.url.path,
            "method": request.method,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
        }
        with open(REQUEST_LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


# El modelo, su versión y el umbral se leen UNA sola vez al iniciar el contenedor
try:
    modelo = mlflow.sklearn.load_model(str(MODEL_DIR))
    with open(MODEL_DIR / "metadata.json") as f:
        meta = json.load(f)
    MODEL_VERSION = str(meta["model_version"])
    DECISION_THRESHOLD = meta["decision_threshold"]
    load_error = None
except Exception as e:
    modelo = None
    MODEL_VERSION = None
    DECISION_THRESHOLD = None
    load_error = str(e)


# ---------------------------------------------------------------
# Datos de entrada
# ---------------------------------------------------------------

class ClientFeatures(BaseModel):
    age: int = Field(alias="age", ge=18, le=100)
    job: str = Field(alias="job")
    marital: str = Field(alias="marital")
    education: str = Field(alias="education")
    default: Literal["yes", "no"] = Field(alias="default")
    balance: int = Field(alias="balance")
    housing: Literal["yes", "no"] = Field(alias="housing")
    loan: Literal["yes", "no"] = Field(alias="loan")
    contact: str = Field(alias="contact")
    day: int = Field(alias="day", ge=1, le=31)
    month: str = Field(alias="month")
    campaign: int = Field(alias="campaign", ge=0)
    pdays: int = Field(alias="pdays", ge=-1)
    previous: int = Field(alias="previous", ge=0)
    poutcome: str = Field(alias="poutcome")

    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------
# Response
# ---------------------------------------------------------------

class PredictionResponse(BaseModel):
    prediction: int
    probability: float | None
    model_version: str


# ---------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------

@app.get("/health")
def health():
    if modelo is None:
        raise HTTPException(status_code=503, detail=f"Modelo no disponible: {load_error}")
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ClientFeatures):
    if modelo is None:
        raise HTTPException(status_code=503, detail="El modelo no se cargó correctamente.")

    data = pd.DataFrame([features.model_dump(by_alias=True)])
    data = lower_case(data)

    # Probabilidad de la clase positiva ("yes") - misma base sobre la que se fijó el umbral
    proba_positiva = float(modelo.predict_proba(data)[0, 1])
    prediction = int(proba_positiva >= DECISION_THRESHOLD)

    return {
        "prediction": prediction,
        "probability": round(proba_positiva, 4),
        "model_version": MODEL_VERSION,
    }
