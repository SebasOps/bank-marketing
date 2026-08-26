"""
Feature engineering para el dataset Bank Marketing.

Este módulo centraliza las transformaciones utilizadas antes del modelado
para garantizar que el procesamiento sea reproducible entre entrenamiento
y producción.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


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


def build_preprocessor():
    """
    Construye el preprocesador utilizado por los modelos de árboles.

    Las variables numéricas se mantienen sin escalamiento, mientras que
    las variables categóricas se convierten mediante One-Hot Encoding.

    Returns
    -------
    ColumnTransformer
        Preprocesador reutilizable para entrenamiento y predicción.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return preprocessor