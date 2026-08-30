# Imports
import sys
import mlflow
import hashlib
from pathlib import Path
import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# --------------------------------------------
# Conectar con el servidor de tracking

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("classification-bank-marketing")

# Ruta raíz del proyecto (cwd = donde se encuentra el notebook; .parent = ruta padre, eso da la ruta raíz)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "bank_marketing.csv"
sys.path.append(str(PROJECT_ROOT))

# Funciones reutilizables
from src.pipelines.split import split_data
from src.features.build_features import feature_selection, encode_target, build_pipeline

# Semilla que se utilizará
RANDOM_SEED = 42

# --------------------------------------------
# Datos 

# Carga de los datos
df = pd.read_csv(PROCESSED_PATH)

# Versión
def get_data_version(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]

DATA_VERSION = get_data_version(PROCESSED_PATH)

# Selección de variables
X = feature_selection(df)
y = encode_target(df['y'])

# Verificar las variables que serán utilizadas como predictoras
print("Variables predictoras:")
print(X.columns.tolist())


# Split
X_train, X_test, y_train, y_test = split_data(X, y)


# --------------------------------------------
# Hiperparámetros a registrar


# --------------------------------------------
# Experimentos

from sklearn.ensemble import RandomForestClassifier
from src.tracking.run_experiment import run_experiment

FEATURE_SET = X.columns.tolist()

# --------------------------------------------
# Random Forest

rf_pipeline = build_pipeline(
    model=RandomForestClassifier(
        random_state=RANDOM_SEED,
        class_weight="balanced",
        max_depth=10,
        n_estimators=100
    ),
    incluir_escalado=False
)

run_experiment(
    pipeline=rf_pipeline,
    model_name="RandomForest",
    params={
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": "balanced",
        "feature_set": FEATURE_SET,
        "random_seed": RANDOM_SEED,
        "data_version": DATA_VERSION,
    },
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    run_name="rf-max_depth_10"
)

