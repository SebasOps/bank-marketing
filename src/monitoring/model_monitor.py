"""
Model Monitoring (O3): calcula Precision_t, Recall_t, F1_t, AUC_t sobre cada
batch de producción (BATCH_1/2/3), usando el modelo ganador exportado en
model_artifact/ (mismo artefacto que sirve la API, sin dependencia de MLflow).

Reutiliza los mismos batches (y el mismo drift) que data_drift.py, para que
O2 y O3 cuenten la misma historia: cuando el PSI se dispara, ¿el modelo
realmente se degrada?
"""

import sys
import json
import numpy as np
import pandas as pd
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.pipelines.split import split_data
from src.features.build_features import feature_selection, encode_target
from src.tracking.config import RANDOM_SEED
from src.monitoring.data_drift import inject_drift

MODEL_DIR = PROJECT_ROOT / "model_artifact"


def compute_model_metrics(y_true, y_pred, y_proba):
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "n_samples": len(y_true),
        "positive_rate_true": round(float(np.mean(y_true)), 4),
        "positive_rate_pred": round(float(np.mean(y_pred)), 4),
    }


if __name__ == "__main__":
    # --------------------------------------------
    # Mismos datos y mismo split que threshold_analysis.py / data_drift.py
    PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "bank_marketing.csv"
    df = pd.read_csv(PROCESSED_PATH)

    X = feature_selection(df)
    y = encode_target(df["y"])
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_test_thr, X_test_final, y_test_thr, y_test_final = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test, random_state=RANDOM_SEED
    )

    X_test_final = X_test_final.reset_index(drop=True)
    y_test_final = pd.Series(y_test_final).reset_index(drop=True)

    # Split estratificado en 3 batches: garantiza misma proporción de clase
    # positiva en cada uno, así el único factor que varía entre batches es
    # el drift inyectado, no la composición base de las clases.
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    X_batches, y_batches = [], []
    for _, batch_idx in skf.split(X_test_final, y_test_final):
        X_batches.append(X_test_final.iloc[batch_idx].reset_index(drop=True))
        y_batches.append(y_test_final.iloc[batch_idx].reset_index(drop=True))

    BATCH_1_X, BATCH_1_y = X_batches[0], y_batches[0]
    BATCH_2_X, BATCH_2_y = inject_drift(X_batches[1], intensity=0.5), y_batches[1]
    BATCH_3_X, BATCH_3_y = inject_drift(X_batches[2], intensity=1.0), y_batches[2]

    # --------------------------------------------
    # Modelo + threshold desde el export final (mismo artefacto que la API)
    modelo = mlflow.sklearn.load_model(str(MODEL_DIR))
    with open(MODEL_DIR / "metadata.json") as f:
        meta = json.load(f)
    DECISION_THRESHOLD = meta["decision_threshold"]

    # --------------------------------------------
    for name, X_batch, y_batch in [
        ("BATCH_1 (sin drift)", BATCH_1_X, BATCH_1_y),
        ("BATCH_2 (drift moderado)", BATCH_2_X, BATCH_2_y),
        ("BATCH_3 (drift fuerte)", BATCH_3_X, BATCH_3_y),
    ]:
        y_proba = modelo.predict_proba(X_batch)[:, 1]
        y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
        metrics = compute_model_metrics(y_batch, y_pred, y_proba)
        print(f"\n=== {name} ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
