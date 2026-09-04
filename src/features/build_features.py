"""
Feature engineering para el dataset Bank Marketing.

Este módulo centraliza las transformaciones utilizadas antes del modelado
para garantizar que el procesamiento sea reproducible entre entrenamiento
y producción.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# Variables numéricas utilizadas por los modelos    # TODO constants.py ??
COLUMNAS_NUMERICAS = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
]

# Variables categóricas utilizadas por los modelos
COLUMNAS_CATEGORICAS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]


# ---------------------------------------------------------------------------------------------------------
# Funciones de selección de características y transformaciones

def feature_selection(df):
    """
    Separa las variables predictoras del dataframe crudo/limpio.
    Excluye 'duration' (data leakage) y 'y' si está presente.
    Se usa tanto en entrenamiento como en producción/inferencia.
    """
    columnas_a_excluir = [col for col in ["y", "duration"] if col in df.columns]
    return df.drop(columns=columnas_a_excluir)


def encode_target(y):
    """
    Codifica la variable objetivo a formato binario (0/1).

    'yes' -> 1, 'no' -> 0. Se centraliza esta conversión para evitar
    inconsistencias entre notebooks y scripts de entrenamiento/evaluación,
    y para que las métricas (F1, Recall, Precision) usen el mismo pos_label
    en todos los cálculos sin necesidad de especificarlo manualmente cada vez.

    Parameters
    ----------
    y : pd.Series
        Variable objetivo original, con valores "yes"/"no".

    Returns
    -------
    pd.Series
        Variable objetivo codificada (1 = yes, 0 = no).
    """
    return y.map({"no": 0, "yes": 1})


def build_preprocessor(incluir_escalado):
    """
    Construye el preprocesador utilizado por los modelos de árboles, por KNN y por
    Regresión Logística.

    - Para los modelos de árboles / incluir_escalado=False
      Las variables numéricas se mantienen sin escalamiento, mientras que
      las variables categóricas se convierten mediante One-Hot Encoding.

    - Para los modelos KNN y RL / incluir_escalado=True
      Las variables numéricas se les aplica escalado con RobustScaler, mientras que
      las variables categóricas se convierten mediante One-Hot Encoding.

    Returns
    -------
    ColumnTransformer
        Preprocesador reutilizable para entrenamiento y predicción.
    """

    transformers = [
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"), COLUMNAS_CATEGORICAS)
    ]

    if incluir_escalado:
        transformers.append(
            ("scaler", RobustScaler(), COLUMNAS_NUMERICAS)
        )
        remainder = "drop"  # todo ya fue transformado explícitamente
    else:
        remainder = "passthrough"  # numéricas pasan sin tocar

    return ColumnTransformer(transformers=transformers, remainder=remainder)


def build_pipeline(model, incluir_escalado, incluir_smote=False, random_seed=42):
    """
    Construye el pipeline completo (preprocesador + resampling opcional + modelo)
    utilizado tanto en entrenamiento como en producción.

    Parameters
    ----------
    model : estimator de sklearn
        Modelo ya instanciado (ej. KNeighborsClassifier(n_neighbors=5)).
    incluir_escalado : bool
        True para modelos sensibles a la escala (KNN, Regresión Logística).
        False para modelos basados en árboles (Decision Tree, Random Forest).
    incluir_smote : bool, default=False
        Si True, agrega SMOTE antes del modelo para balancear las clases.
        Solo se ejecuta durante el entrenamiento (.fit()); no afecta .predict().
        Pensado para modelos sin soporte nativo de class_weight, como KNN.
    random_seed : int, default=42
        Semilla utilizada por SMOTE para garantizar reproducibilidad.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Pipeline reutilizable para entrenamiento y predicción.
    """

    preprocessor = build_preprocessor(incluir_escalado)

    steps = [("preprocessor", preprocessor)]

    if incluir_smote:
        steps.append(("smote", SMOTE(random_state=random_seed)))

    steps.append(("model", model))

    return ImbPipeline(steps)
