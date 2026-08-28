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


# Variables numéricas utilizadas por los modelos
NUMERIC_FEATURES = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
]

# Variables categóricas utilizadas por los modelos
CATEGORICAL_FEATURES = [
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
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_FEATURES)
    ]

    if incluir_escalado:
        transformers.append(
            ("scaler", RobustScaler(), NUMERIC_FEATURES)
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