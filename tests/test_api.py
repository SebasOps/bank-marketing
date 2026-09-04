"""
Pruebas sobre la API (FastAPI), sin necesidad de Docker: se prueba la app
directamente en memoria con TestClient.

Cubre: request válido -> HTTP 200 -> schema de respuesta válido,
       y qué pasa frente a distintos tipos de input inválido.

Correr con: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

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


@pytest.fixture(scope="module", autouse=True)
def verificar_modelo_cargado():
    """Si el modelo no está disponible, se saltan las pruebas que dependen de
    una predicción real, en vez de fallar en rojo por un problema de setup."""
    resp = client.get("/health")
    if resp.status_code != 200:
        pytest.skip("El modelo no está cargado (¿corriste export_model.py?)")


# ---------------------------------------------------------------
# CASO FELIZ: request válido -> 200 -> schema válido
# ---------------------------------------------------------------

def test_health_responde_200():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_con_input_valido_responde_200():
    resp = client.post("/predict", json=INPUT_VALIDO)
    assert resp.status_code == 200


def test_predict_respeta_el_schema_de_respuesta():
    resp = client.post("/predict", json=INPUT_VALIDO)
    body = resp.json()
    assert set(body.keys()) == {"prediction", "probability", "model_version"}
    assert isinstance(body["prediction"], int)
    assert isinstance(body["model_version"], str)
    assert body["probability"] is None or isinstance(body["probability"], float)


def test_prediccion_es_clase_valida():
    resp = client.post("/predict", json=INPUT_VALIDO)
    assert resp.json()["prediction"] in {0, 1}


def test_probabilidad_en_rango_valido():
    resp = client.post("/predict", json=INPUT_VALIDO)
    probability = resp.json()["probability"]
    if probability is not None:
        assert 0.0 <= probability <= 1.0


# ---------------------------------------------------------------
# Input inválido 
# ---------------------------------------------------------------

def test_falta_una_variable_obligatoria():
    incompleto = INPUT_VALIDO.copy()
    del incompleto["age"]
    resp = client.post("/predict", json=incompleto)
    assert resp.status_code == 422
    assert "detail" in resp.json()


def test_tipo_de_dato_incorrecto():
    invalido = INPUT_VALIDO.copy()
    invalido["age"] = "treinta"
    resp = client.post("/predict", json=invalido)
    assert resp.status_code == 422


def test_edad_fuera_de_rango_menor():
    invalido = INPUT_VALIDO.copy()
    invalido["age"] = 10
    resp = client.post("/predict", json=invalido)
    assert resp.status_code == 422


def test_edad_fuera_de_rango_mayor():
    invalido = INPUT_VALIDO.copy()
    invalido["age"] = 150
    resp = client.post("/predict", json=invalido)
    assert resp.status_code == 422


def test_day_fuera_de_rango():
    invalido = INPUT_VALIDO.copy()
    invalido["day"] = 32
    resp = client.post("/predict", json=invalido)
    assert resp.status_code == 422


def test_campaign_negativo():
    invalido = INPUT_VALIDO.copy()
    invalido["campaign"] = -1
    resp = client.post("/predict", json=invalido)
    assert resp.status_code == 422


def test_pdays_menor_a_menos_uno():
    invalido = INPUT_VALIDO.copy()
    invalido["pdays"] = -5
    resp = client.post("/predict", json=invalido)
    assert resp.status_code == 422


def test_previous_negativo():
    invalido = INPUT_VALIDO.copy()
    invalido["previous"] = -1
    resp = client.post("/predict", json=invalido)
    assert resp.status_code == 422


def test_default_valor_no_valido():
    """'default' solo acepta 'yes'/'no', no cualquier string."""
    invalido = INPUT_VALIDO.copy()
    invalido["default"] = "tal_vez"
    resp = client.post("/predict", json=invalido)
    assert resp.status_code == 422


def test_body_vacio():
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_mensaje_de_error_es_informativo():
    """El error 422 debe indicar QUÉ campo falló, no solo que algo falló."""
    invalido = INPUT_VALIDO.copy()
    invalido["age"] = 150
    resp = client.post("/predict", json=invalido)
    detalle = resp.json()["detail"]
    campos_reportados = [str(err.get("loc")) for err in detalle]
    assert any("age" in campo for campo in campos_reportados)
