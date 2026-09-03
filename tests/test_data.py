"""
Pruebas sobre los DATOS de entrada (dataset Bank Marketing, ya limpio).

Cubre: esquema, tipos, rangos, missing, variables obligatorias.
Reutiliza las funciones reales de src/quality/clean.py y src/quality/gates.py
en vez de reinventar las reglas de validación y causar duplicidad.

Correr con: pytest tests/test_data.py -v
"""

# Imports
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.quality.clean import (
    COLUMNAS_CATEGORICAS,
    COLUMNAS_NUMERICAS,
    rename_columns,
    lower_case,
    resolve_types,
)
from src.quality.gates import (
    check_minimum_rows,
    check_target_no_nulls,
    check_impossible_data,
    expected_columns,
    expected_target_classes,
    data_quality_gates,
)

# Ruta del dataset procesado
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "bank_marketing.csv"
COLUMNAS_ESPERADAS = set(COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS)


@pytest.fixture(scope="module")
def df():
    """Carga y limpia el dataset real, una sola vez para todas las pruebas."""
    if not PROCESSED_PATH.exists():
        pytest.skip(f"No existe {PROCESSED_PATH}. Corre primero el pipeline de ingesta.")
    raw = pd.read_csv(PROCESSED_PATH)
    cleaned = rename_columns(raw)
    cleaned = lower_case(cleaned)
    cleaned = resolve_types(cleaned)
    return cleaned


# ---------------------------------------------------------------
# Variables obligatorio
# ---------------------------------------------------------------

def test_columnas_esperadas_presentes(df):
    """Las funciones reales de gates deben validar sin lanzar excepción."""
    expected_columns(df)


def test_no_hay_columnas_extra_ni_faltantes(df):
    assert set(df.columns) == COLUMNAS_ESPERADAS


@pytest.mark.parametrize("col", sorted(COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS))
def test_variable_obligatoria_presente(df, col):
    assert col in df.columns


# ---------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------

def test_columnas_numericas_son_numericas(df):
    for col in COLUMNAS_NUMERICAS:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} no es numérica"


def test_columnas_categoricas_son_texto(df):
    for col in COLUMNAS_CATEGORICAS:
        assert pd.api.types.is_object_dtype(df[col]), f"{col} no es texto"


# ---------------------------------------------------------------
# Missing Values
# ---------------------------------------------------------------

def test_target_sin_nulos(df):
    check_target_no_nulls(df, target="y")


def test_sin_infinitos_en_numericas(df):
    assert np.isfinite(df[COLUMNAS_NUMERICAS].values).all()


# ---------------------------------------------------------------
# Rangos y Datos disponibles
# ---------------------------------------------------------------

def test_sin_valores_imposibles(df):
    check_impossible_data(df)


def test_clases_target_esperadas(df):
    expected_target_classes(df)


def test_minimo_de_filas(df):
    check_minimum_rows(df, minimum_rows=5000)


# ---------------------------------------------------------------
# Los gates deben rechazar los datos malos y solo aceptar los buenos
# ---------------------------------------------------------------

def test_gate_rechaza_edad_imposible(df):
    corrupto = df.copy()
    corrupto.loc[corrupto.index[0], "age"] = 150
    with pytest.raises(AssertionError):
        check_impossible_data(corrupto)


def test_gate_rechaza_campaign_negativo(df):
    corrupto = df.copy()
    corrupto.loc[corrupto.index[0], "campaign"] = -1
    with pytest.raises(AssertionError):
        check_impossible_data(corrupto)


def test_gate_rechaza_previous_negativo(df):
    corrupto = df.copy()
    corrupto.loc[corrupto.index[0], "previous"] = -1
    with pytest.raises(AssertionError):
        check_impossible_data(corrupto)


def test_gate_rechaza_target_con_nulos(df):
    corrupto = df.copy()
    corrupto.loc[corrupto.index[0], "y"] = None
    with pytest.raises(AssertionError):
        check_target_no_nulls(corrupto, target="y")


def test_gate_rechaza_clase_no_esperada_en_target(df):
    corrupto = df.copy()
    corrupto.loc[corrupto.index[0], "y"] = "maybe"
    with pytest.raises(AssertionError):
        expected_target_classes(corrupto)


def test_gate_rechaza_columna_faltante(df):
    corrupto = df.drop(columns=["age"])
    with pytest.raises(AssertionError):
        expected_columns(corrupto)


def test_gate_rechaza_dataset_muy_pequenio(df):
    corrupto = df.sample(n=10, random_state=0)
    with pytest.raises(AssertionError):
        check_minimum_rows(corrupto, minimum_rows=5000)


# ---------------------------------------------------------------
# Punto de entrada unico
# ---------------------------------------------------------------

def test_data_quality_gates_pasa_con_datos_reales(df):
    """El punto de entrada único debe correr sin lanzar excepción sobre datos limpios reales."""
    resultado = data_quality_gates(df)
    assert resultado is not None
