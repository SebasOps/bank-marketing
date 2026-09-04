# --------------------------------------------
# Imports
import sys
import mlflow
import hashlib
from pathlib import Path
import pandas as pd
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score

# Calcular ruta raíz y agregarla a sys.path para poder importar desde src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))   

from src.quality.gates import data_quality_gates
from src.pipelines.split import split_data
from src.features.build_features import feature_selection, encode_target, build_pipeline
from src.tracking.config import RANDOM_SEED, CLASS_BALANCED, INCLUIR_SMOTE_KNN, NO_INCLUIR_SMOTE_KNN, INCLUIR_ESCALADO, NO_INCLUIR_ESCALADO, get_data_version
from src.tracking.run_experiment import run_experiment

# --------------------------------------------
# Conectar con el servidor de mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("classification-bank-marketing")


# --------------------------------------------
# Datos 

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "bank_marketing.csv"
df = pd.read_csv(PROCESSED_PATH)

# Data Quality Gates: valida el dataset antes de cualquier paso de
# entrenamiento. Si alguna regla falla, detiene el pipeline aquí
df = data_quality_gates(df)

# Versión
DATA_VERSION = get_data_version(PROCESSED_PATH)

# Selección de variables
X = feature_selection(df)
y = encode_target(df['y'])

# Verificar las variables que serán utilizadas como predictoras
FEATURE_SET = X.columns.tolist()

print("Variables predictoras:")
print(FEATURE_SET)

# Split
X_train, X_test, y_train, y_test = split_data(X, y)


# --------------------------------------------
# Experimentos - Random Forest

from sklearn.ensemble import RandomForestClassifier

N_ESTIMATORS_RF = 100
VALORES_MAX_DEPTH_RF = [5, 10, 15]

for max_depth in VALORES_MAX_DEPTH_RF:
    rf_pipeline = build_pipeline(
        model=RandomForestClassifier(
            random_state=RANDOM_SEED,
            class_weight=CLASS_BALANCED,
            max_depth=max_depth,
            n_estimators=N_ESTIMATORS_RF
        ),
        incluir_escalado=NO_INCLUIR_ESCALADO
    )

    run_experiment(
        pipeline=rf_pipeline,
        model_name="RandomForest",
        params={
            "n_estimators": N_ESTIMATORS_RF,
            "max_depth": max_depth,
            "class_weight": CLASS_BALANCED,
            "feature_set": FEATURE_SET,
            "random_seed": RANDOM_SEED,
            "data_version": DATA_VERSION,
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        run_name=f"rf-max_depth_{max_depth}"
    )


MAX_DEPTH_GANADOR = 10  # en base al experimento anterior
MIN_SAMPLE_LEAF = [5, 10, 15]

for min_samples_leaf in MIN_SAMPLE_LEAF:
    rf_pipeline = build_pipeline(
        model=RandomForestClassifier(
            random_state=RANDOM_SEED,
            class_weight=CLASS_BALANCED,
            max_depth=MAX_DEPTH_GANADOR,
            n_estimators=N_ESTIMATORS_RF,
            min_samples_leaf=min_samples_leaf
        ),
        incluir_escalado=NO_INCLUIR_ESCALADO
    )

    run_experiment(
        pipeline=rf_pipeline,
        model_name="RandomForest",
        params={
            "n_estimators": N_ESTIMATORS_RF,
            "max_depth": MAX_DEPTH_GANADOR,
            "min_samples_leaf": min_samples_leaf,
            "class_weight": CLASS_BALANCED,
            "feature_set": FEATURE_SET,
            "random_seed": RANDOM_SEED,
            "data_version": DATA_VERSION,
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        run_name=f"rf-max_depth_{MAX_DEPTH_GANADOR}_estimators_{N_ESTIMATORS_RF}_min_samples_leaf_{min_samples_leaf}"
    )


# --------------------------------------------
# Experimentos - Decision Tree

from sklearn.tree import DecisionTreeClassifier

VALORES_MAX_DEPTH_DT = [10, 15]

for max_depth in VALORES_MAX_DEPTH_DT:
    dt_pipeline = build_pipeline(
        model=DecisionTreeClassifier(
            random_state=RANDOM_SEED,
            class_weight=CLASS_BALANCED,
            max_depth=max_depth
        ),
        incluir_escalado=NO_INCLUIR_ESCALADO
    )

    run_experiment(
        pipeline=dt_pipeline,
        model_name="DecisionTree",
        params={
            "max_depth": max_depth,
            "class_weight": CLASS_BALANCED,
            "feature_set": FEATURE_SET,
            "random_seed": RANDOM_SEED,
            "data_version": DATA_VERSION,
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        run_name=f"dt-max_depth_{max_depth}"
    )


# --------------------------------------------
# Experimentos - KNN

from sklearn.neighbors import KNeighborsClassifier

"""
Duante el feature-engineering se decidió usar los valores
15 y 33 para k.
"""

VALORES_K = [15, 31]

for k in VALORES_K:
    knn_pipeline = build_pipeline(
        model=KNeighborsClassifier(n_neighbors=k),
        incluir_escalado=INCLUIR_ESCALADO,
        incluir_smote=INCLUIR_SMOTE_KNN,
        random_seed=RANDOM_SEED
    )

    run_experiment(
        pipeline=knn_pipeline,
        model_name="KNN",
        params={
            "n_neighbors": k,
            "smote": INCLUIR_SMOTE_KNN,
            "feature_set": FEATURE_SET,
            "random_seed": RANDOM_SEED,
            "data_version": DATA_VERSION,
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        run_name=f"knn-k_{k}-smote_{INCLUIR_SMOTE_KNN}"
    )

"""
Observando los resultados de las métricas de los experimentos
de KNN, se considera mejor el experimento que usó k=33, 
a pesar de que los resultados no son muy convincentes se 
probará el modelo con incluir_smote=NO_INCLUIR_SMOTE_KNN 
(no se aplica el balanceo SMOTE).
"""

VALORES_K = 31

knn_pipeline_2 = build_pipeline(
    model=KNeighborsClassifier(n_neighbors=VALORES_K),
    incluir_escalado=INCLUIR_ESCALADO,
    incluir_smote=NO_INCLUIR_SMOTE_KNN,
    random_seed=RANDOM_SEED
)

run_experiment(
    pipeline=knn_pipeline_2,
    model_name="KNN",
    params={
        "n_neighbors": VALORES_K,
        "smote": NO_INCLUIR_SMOTE_KNN,
        "feature_set": FEATURE_SET,
        "random_seed": RANDOM_SEED,
        "data_version": DATA_VERSION,
    },
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    run_name=f"knn-k_{VALORES_K}-smote_{NO_INCLUIR_SMOTE_KNN}"
)


# --------------------------------------------
# Experimentos - Regresión Logística

from sklearn.linear_model import LogisticRegression

VALORES_C_RL = [0.4, 0.6] 

for C in VALORES_C_RL:
    rl_pipeline = build_pipeline(
        model=LogisticRegression(
            random_state=RANDOM_SEED,
            class_weight=CLASS_BALANCED,
            max_iter=1000,
            C=C
        ),
        incluir_escalado=INCLUIR_ESCALADO
    )

    run_experiment(
        pipeline=rl_pipeline,
        model_name="LogisticRegression",
        params={
            "C": C,
            "class_weight": CLASS_BALANCED,
            "max_iter": 1000,
            "feature_set": FEATURE_SET,
            "random_seed": RANDOM_SEED,
            "data_version": DATA_VERSION,
        },
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        run_name=f"rl-C_{C}"
    )
