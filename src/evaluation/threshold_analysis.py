"""
Análisis de umbral de decisión para Random Forest (modelo ganador)

Por defecto, un modelo de clasificación binaria asigna la clase positiva
cuando su probabilidad predicha supera 0.5, pero ese corte es arbitrario:
no proviene del entrenamiento ni está optimizado para el objetivo del
negocio. Este análisis carga el modelo ganador ya entrenado (sin
reentrenar) desde su run en MLflow, y evalúa cómo cambian recall,
precisión y F1 al mover ese umbral, para entender el trade-off entre
detectar más clientes con intención de conversión y la cantidad de
contactos erróneos que eso implica.
"""

# --------------------------------------------
# Imports

import sys
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd

# Calcular ruta raíz y agregarla a sys.path para poder importar desde src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))   

from src.pipelines.split import split_data
from src.features.build_features import feature_selection, encode_target
from src.tracking.config import RANDOM_SEED, CLASS_BALANCED, get_data_version
from src.tracking.run_experiment import run_experiment


# --------------------------------------------
# Conectar con el servidor de mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("classification-bank-marketing")

RUN_ID_GANADOR = "c027c83439f64e46a3f91f1f12ea3088"


# --------------------------------------------
# Mismos datos que se usó en training.py 

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "bank_marketing.csv"
df = pd.read_csv(PROCESSED_PATH)

# Selección de variables
X = feature_selection(df)
y = encode_target(df["y"])

# Split
X_train, X_test, y_train, y_test = split_data(X, y)

# Versión
DATA_VERSION = get_data_version(PROCESSED_PATH)

# Variables predictoras
FEATURE_SET = X.columns.tolist()

# max_depth del modelo ganador 
MAX_DEPTH_GANADOR = 10 

# n_estimators del modelo ganador
N_ESTIMATORS_GANADOR = 100

# Umbrales a analizar
UMBRALES = [0.45, 0.4]

# Pipeline del modelo ganador
rf_pipeline = mlflow.sklearn.load_model(f"runs:/{RUN_ID_GANADOR}/model")

# predict_proba solo se calcula una vez, el umbral se aplica después sobre las mismas probabilidades
y_proba = rf_pipeline.predict_proba(X_test)[:, 1]


# --------------------------------------------
# Experimentos - Modelo ganador 

for umbral in UMBRALES:
    run_experiment(
        pipeline=rf_pipeline,
        model_name="RandomForest",
        params={
            "max_depth": MAX_DEPTH_GANADOR, 
            "n_estimators": N_ESTIMATORS_GANADOR, 
            "class_weight": CLASS_BALANCED,
            "random_seed": RANDOM_SEED,
            "feature_set": FEATURE_SET,
            "data_version": DATA_VERSION,
            },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        run_name=f"rf-threshold_{umbral}",
        already_fitted=True,
        threshold=umbral
    )
