"""
Funciones de limpieza del dataset bank-marketing.

Se centraliza toda la lógica de limpieza en este módulo para que exista una única
fuente de verdad: el notebook y cualquier script de producción importan estas mismas
funciones en lugar de duplicarlas, evitando que ambas versiones se desincronicen.

Nota. Algunas de las funciones de limpieza se hicieron con el fin de automatizar, aunque
      dichos errores no se hayan visto en el diagnóstico del Data Quality. No se implementaron
      limpieza de nulos o de datos imposibles ya que no sería correcto decidir que hacer
      con ellos antes de tratar con un caso real.
"""

# Imports
import pandas as pd
from pathlib import Path


# Constantes
COLUMNAS_CATEGORICAS = [
    "job", "marital", "education", "default",
    "housing", "loan", "contact", "month",
    "poutcome", "y"
]

COLUMNAS_NUMERICAS = [
    "age", "balance", "day", "duration",
    "campaign", "pdays", "previous"
]


# ---------------------------------------------------------------------------------------------------------
# Funciones de limpieza

def rename_columns(df):
    """
    Renombra day_of_week -> day (representa día del mes, no día de la semana).
    """
    df = df.copy() # No modifica el df original

    return df.rename(columns={"day_of_week": "day"})


def drop_duplicates(df):
    """
    Elimina filas completamente duplicadas.
    """
    df = df.copy()

    return df.drop_duplicates()


def lower_case(df):
    """
    Normaliza texto en columnas categóricas: minúsculas y sin espacios al inicio/final.
    """
    df = df.copy()

    for col in COLUMNAS_CATEGORICAS:
        df[col] = df[col].str.lower()
        df[col] = df[col].str.strip()

    return df


def resolve_types(df):
    """
    Convierte a numérico las columnas que deberían serlo, en caso de que hayan llegado como texto.
    """
    df = df.copy()

    for col in COLUMNAS_NUMERICAS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------------------------------------
# Función que corre todas las funciones de limpieza (reproducicle)

def clean_data(df):
    df = rename_columns(df)
    df = drop_duplicates(df)
    df = lower_case(df)
    df = resolve_types(df)

    return df


# ---------------------------------------------------------------------------------------------------------
# Función que corre solo cuando se corre directamente este .py (python src/quality/clean.py)

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    RAW_PATH = PROJECT_ROOT / "data" / "raw" / "bank_marketing.csv"

    df = pd.read_csv(RAW_PATH)
    df_limpio = clean_data(df)

    # Carpeta donde se almacena los datos procesados
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Ruta de salida
    output_path = PROCESSED_DIR / "bank_marketing.csv"

    # Guardar dataset
    df_limpio.to_csv(output_path, index=False)

    print(f"Dataset limpio guardado en: {output_path}")
