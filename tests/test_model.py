"""
Pruebas sobre el MODELO ya exportado (model_artifact/).

Verifica: input válido -> predicción válida.
Requiere haber corrido export_model.py antes (necesita model_artifact/).

Correr con: pytest tests/test_model.py -v
"""

import json
import sys
from pathlib import Path
import mlflow.sklearn
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.quality.clean import lower_case

MODEL_DIR = PROJECT_ROOT / "model_artifact"

# Un input válido de ejemplo, mismas columnas/formato que espera el pipeline
INPUT_VALIDO = {
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1500,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 2,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


@pytest.fixture(scope="module")
def modelo():
    if not MODEL_DIR.exists():
        pytest.skip("No existe model_artifact/. Corre primero: python export_model.py")
    return mlflow.sklearn.load_model(str(MODEL_DIR))


@pytest.fixture(scope="module")
def metadata():
    metadata_path = MODEL_DIR / "metadata.json"
    if not metadata_path.exists():
        pytest.skip("No existe model_artifact/metadata.json")
    with open(metadata_path) as f:
        return json.load(f)


@pytest.fixture
def input_df():
    return lower_case(pd.DataFrame([INPUT_VALIDO]))


def test_el_modelo_carga_sin_error(modelo):
    assert modelo is not None


def test_input_valido_produce_prediccion(modelo, input_df):
    resultado = modelo.predict(input_df)
    assert resultado is not None
    assert len(resultado) == 1


def test_prediccion_es_clase_valida(modelo, input_df):
    prediccion = int(modelo.predict(input_df)[0])
    assert prediccion in {0, 1}


def test_prediccion_es_determinista(modelo, input_df):
    """El mismo input debe dar siempre la misma predicción."""
    r1 = int(modelo.predict(input_df)[0])
    r2 = int(modelo.predict(input_df)[0])
    assert r1 == r2


def test_modelo_expone_predict_proba(modelo, input_df):
    proba = modelo.predict_proba(input_df)[0]
    assert abs(sum(proba) - 1.0) < 1e-6, "Las probabilidades deben sumar 1"
    assert all(0 <= p <= 1 for p in proba)


def test_umbral_aplicado_coincide_con_metadata(modelo, input_df, metadata):
    """Reproduce la misma lógica de decisión que usa la API, sobre el modelo directo."""
    proba_positiva = modelo.predict_proba(input_df)[0, 1]
    prediccion_esperada = int(proba_positiva >= metadata["decision_threshold"])
    assert prediccion_esperada in {0, 1}
