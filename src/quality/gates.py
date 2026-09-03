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

    if "age" in df.columns:
        n_age = ((df["age"] < 18) | (df["age"] > 100)).sum()
        assert n_age == 0, (
            f"La variable 'age' presenta {n_age} valores imposibles"
        )

    if "duration" in df.columns:
        n_duration = (df["duration"] < 0).sum()
        assert n_duration == 0, (
            f"La variable 'duration' presenta {n_duration} valores imposibles (negativos)"
        )

    if "campaign" in df.columns:
        n_campaign = (df["campaign"] < 1).sum()
        assert n_campaign == 0, (
            f"La variable 'campaign' presenta {n_campaign} valores imposibles (menores a 1)"
        )

    if "previous" in df.columns:
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
# Gates adicionales para batches de producción (inferencia)

def expected_feature_columns(df, reference_df):
    """
    Valida que un batch de producción tenga exactamente las mismas columnas
    que reference_df (ej. X_train). A diferencia de expected_columns(), no
    depende de las constantes de clean.py (que incluyen 'y' y 'duration',
    ausentes en tiempo de inferencia), sino del feature set real del modelo.
    """
    columnas_actuales = set(df.columns)
    columnas_esperadas = set(reference_df.columns)

    no_esperadas = columnas_actuales - columnas_esperadas
    faltantes = columnas_esperadas - columnas_actuales

    mensaje = ""
    if no_esperadas:
        mensaje += f"Columnas no esperadas: {no_esperadas}. "
    if faltantes:
        mensaje += f"Columnas faltantes: {faltantes}."

    assert not no_esperadas and not faltantes, mensaje


def check_no_missing_values(df):
    """Verifica que no existan valores nulos en ninguna columna del batch."""
    nulos = df.isna().sum()
    con_nulos = nulos[nulos > 0]

    assert len(con_nulos) == 0, (
        f"Se detectaron valores nulos inesperados: {con_nulos.to_dict()}"
    )


def check_no_duplicates(df):
    """Verifica que no existan filas completamente duplicadas en el batch."""
    n_duplicados = df.duplicated().sum()

    assert n_duplicados == 0, (
        f"El batch presenta {n_duplicados} filas duplicadas"
    )


def check_no_extreme_outliers(df, reference_df, iqr_multiplier=3):
    """
    Verifica outliers extremos en columnas numéricas, con límites calculados
    dinámicamente sobre reference_df (rango intercuartílico x3, más laxo que
    el estándar 1.5 para capturar solo valores verdaderamente extremos, no
    cualquier variación normal de muestreo).
    """
    numeric_cols = reference_df.select_dtypes(include="number").columns
    errores = {}

    for col in numeric_cols:
        if col not in df.columns:
            continue
        q1, q3 = reference_df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        n_fuera = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_fuera > 0:
            errores[col] = f"{n_fuera} valores fuera de [{lo:.1f}, {hi:.1f}]"

    assert len(errores) == 0, (
        f"Se detectaron outliers extremos: {errores}"
    )


def check_valid_categories(df, reference_df):
    """
    Verifica que las columnas categóricas no presenten valores nunca vistos
    en reference_df. Se compara contra el propio dataset de referencia (no
    contra una lista hardcodeada) para no asumir categorías incorrectas.
    """
    categorical_cols = reference_df.select_dtypes(exclude="number").columns
    errores = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue
        conocidas = set(reference_df[col].dropna().unique())
        actuales = set(df[col].dropna().unique())
        no_esperadas = actuales - conocidas
        if no_esperadas:
            errores[col] = list(no_esperadas)

    assert len(errores) == 0, (
        f"Se detectaron categorías no vistas en referencia: {errores}"
    )


# ---------------------------------------------------------------------------------------------------------
# Funciones que corren las funciones de gates, ya sea para validación o producción (reproducicle)

def data_quality_gates(df):
    """Gates para dataset completo de entrenamiento."""
    check_minimum_rows(df, minimum_rows=5000)
    check_target_no_nulls(df, target="y")
    check_impossible_data(df)
    expected_columns(df)
    expected_target_classes(df)

    return df


def production_quality_gates(df, reference_df):
    """
    Gates para batches de producción (inferencia).
    A diferencia de data_quality_gates() (dataset completo de 
    entrenamiento), no exige un mínimo de filas ni valida 'y' 
    El batch de producción no tiene target ni garantía de tamaño mínimo.
    """
    
    expected_feature_columns(df, reference_df)
    check_impossible_data(df)
    check_no_missing_values(df)
    check_no_duplicates(df)
    check_no_extreme_outliers(df, reference_df)
    check_valid_categories(df, reference_df)

    return df