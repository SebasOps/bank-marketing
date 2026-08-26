"""
Data Quality Gates para el dataset bank-marketing.

Validaciones automáticas que se ejecutan antes de entrenar el modelo, sobre el
dataset ya limpio. A diferencia de src/cleaning/clean.py, estas funciones no
corrigen ni transforman datos: solo verifican que se cumplan las condiciones
mínimas de calidad, y detienen el pipeline (AssertionError) si alguna falla.

Se agrupan todas en data_quality_gates(df), el punto de entrada que las ejecuta
en orden y que debe correrse antes de cualquier etapa de entrenamiento.
"""

# ---------------------------------------------------------------------------------------------------------
# Funciones gates

def check_minimum_rows(df, minimum_rows=5000):
    """El dataset debe tener al menos minimum_rows registros."""

    assert df.shape[0] >= minimum_rows, (
        f"Dataset tiene {df.shape[0]} filas, se esperaban al menos {minimum_rows}"
    )


def check_target_no_nulls(df, target="y"):
    """La variable objetivo no puede tener valores nulos."""

    n_nulls = df[target].isna().sum()
    assert n_nulls == 0, (
        f"La variable objetivo '{target}' tiene {n_nulls} valores nulos"
    )


def check_impossible_data(df):
    """ Verifica que las variables numéricas no contengan valores fuera de sus rangos posibles."""
    
    n_age = ((df["age"] < 18) | (df["age"] > 100)).sum()
    assert n_age == 0, (
        f"La variable 'age' presenta {n_age} valores imposibles"
    )

    n_duration = (df["duration"] < 0).sum()                                                 # TODO Revisar si mantener la validación de 'duration'
    assert n_duration == 0, (
        f"La variable 'duration' presenta {n_duration} valores imposibles (negativos)"
    )

    n_campaign = (df["campaign"] < 1).sum()
    assert n_campaign == 0, (
        f"La variable 'campaign' presenta {n_campaign} valores imposibles (menores a 1)"
    )

    n_previous = (df["previous"] < 0).sum()
    assert n_previous == 0, (
        f"La variable 'previous' presenta {n_previous} valores imposibles (negativos)"
    )


def expected_columns(df):
    """Valida que las columnas que trae el dataset sean las esperadas"""

    # Constantes traídas de src/quality/clean.py para no duplicar lógica
    from .clean import COLUMNAS_CATEGORICAS, COLUMNAS_NUMERICAS
    COLUMNAS_ESPERADAS = set(COLUMNAS_CATEGORICAS + COLUMNAS_NUMERICAS)

    columnas_actuales = set(df.columns) # Columnas que trae el df a revisar

    # Revisión 
    columnas_no_esperadas = columnas_actuales - COLUMNAS_ESPERADAS
    columnas_faltantes = COLUMNAS_ESPERADAS - columnas_actuales

    n_errores = len(columnas_no_esperadas) + len(columnas_faltantes)

    # Mensaje
    if len(columnas_no_esperadas) > 0:
        mensaje_no_esperadas = f"El dataset presenta las siguientes columnas no esperadas: {columnas_no_esperadas}. "
    else:
        mensaje_no_esperadas = ""
    
    if len(columnas_faltantes) > 0:
        mensaje_faltantes = f"El dataset no presenta las siguienres columnas esperadas: {columnas_faltantes}"
    else:
        mensaje_faltantes = ""

    mensaje = mensaje_no_esperadas + mensaje_faltantes

    assert n_errores == 0, (
        mensaje
    )


def expected_target_classes(df):
    """Valida que la variable target 'y' no tenga clases no esperadas"""

    # Constantes
    CLASES_ESPERADAS = set(["yes", "no"])
    clases_actuales = set(df['y'].values)

    # Revisón
    clases_no_esperadas = clases_actuales - CLASES_ESPERADAS

    n_clases_no_esperadas = len(clases_no_esperadas)

    assert n_clases_no_esperadas == 0, (
        f"La variable target 'y' presenta las siguientes clases no esperadas: {clases_no_esperadas}."
    )


# ---------------------------------------------------------------------------------------------------------
# Función que corre todas las funciones de gates (reproducicle)

def data_quality_gates(df):
    check_minimum_rows(df, minimum_rows=5000)
    check_target_no_nulls(df, target="y")
    check_impossible_data(df)
    expected_columns(df)
    expected_target_classes(df)

    return df
    