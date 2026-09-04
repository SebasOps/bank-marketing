"""
Estrategia de reentrenamiento (trigger).

Combina dos señales independientes para decidir si se dispara el pipeline
de reentrenamiento:
    1. Data drift (PSI): ¿cambió la distribución de los datos de entrada?
    2. Model performance (PR-AUC): ¿se degradó la calidad del modelo?

Justificación de por qué se requieren AMBAS señales, no solo PSI:
Drift alto no implica degradación del modelo -- puede ocurrir en una
dirección que el modelo ya sabe manejar (ver BATCH_2: PSI en ALERT en
'balance'/'job', pero PR-AUC sube de 0.47 a 0.52). Reentrenar solo por PSI
alto desperdiciaría cómputo y arriesgaría producir un modelo peor (menos
datos limpios disponibles, riesgo de sobreajustar a ruido reciente), sin
ninguna mejora real. A la inversa, un modelo puede degradarse sin drift
visible en las features (concept drift: cambia la relación X->y, que PSI
no puede ver porque solo compara P(X), nunca P(y|X)) -- en ese caso, no
corresponde reentrenar automáticamente sin investigación humana, porque
la causa raíz podría no resolverse simplemente con más datos recientes.
"""

# Imports
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split, StratifiedKFold
import mlflow.sklearn

# Raíz del proyecto 
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.pipelines.split import split_data
from src.features.build_features import feature_selection, encode_target
from src.tracking.config import RANDOM_SEED
from src.monitoring.data_drift import compute_drift_report, inject_drift
from src.monitoring.model_monitor import compute_model_metrics


MODEL_DIR = PROJECT_ROOT / "model_artifact"
LOG_PATH = PROJECT_ROOT / "logs" / "retraining_decisions.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)


# Thresholds 
PSI_ALERT_THRESHOLD = 0.25   # estándar de industria (scoring crediticio)
PR_AUC_BASELINE = 0.46       # PR-AUC del holdout final (rf-final-holdout)
PR_AUC_FLOOR = 0.40          # ~13% de caída relativa tolerada antes de considerar degradación real


def decide_retraining(psi_report: dict, pr_auc: float,
                       psi_threshold=PSI_ALERT_THRESHOLD, pr_auc_floor=PR_AUC_FLOOR):
    features_en_alerta = [f for f, r in psi_report.items() if r["psi"] > psi_threshold]
    drift_detectado = len(features_en_alerta) > 0
    degradacion_detectada = pr_auc < pr_auc_floor
    trigger = drift_detectado and degradacion_detectada

    if trigger:
        razon = "Drift Y degradación detectados simultáneamente -> reentrenar"
    elif drift_detectado:
        razon = "Drift detectado pero el modelo sigue siendo confiable -> NO reentrenar, solo monitorear"
    elif degradacion_detectada:
        razon = "Degradación sin drift visible en features -> posible concept drift, investigar manualmente (no auto-reentrenar)"
    else:
        razon = "Sin drift ni degradación -> sistema estable"

    return {
        "drift_detectado": drift_detectado,
        "features_en_alerta": features_en_alerta,
        "pr_auc": round(pr_auc, 4),
        "degradacion_detectada": degradacion_detectada,
        "trigger_retraining": trigger,
        "razon": razon,
    }


if __name__ == "__main__":
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

    REFERENCE = X_train.reset_index(drop=True)

    # Mismos batches estratificados que model_monitor.py
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    X_batches, y_batches = [], []
    for _, idx in skf.split(X_test_final, y_test_final):
        X_batches.append(X_test_final.iloc[idx].reset_index(drop=True))
        y_batches.append(y_test_final.iloc[idx].reset_index(drop=True))

    BATCH_1_X, BATCH_1_y = X_batches[0], y_batches[0]
    BATCH_2_X, BATCH_2_y = inject_drift(X_batches[1], intensity=0.5), y_batches[1]
    BATCH_3_X, BATCH_3_y = inject_drift(X_batches[2], intensity=1.0), y_batches[2]

    numeric_cols = REFERENCE.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = REFERENCE.select_dtypes(exclude=[np.number]).columns.tolist()

    modelo = mlflow.sklearn.load_model(str(MODEL_DIR))
    with open(MODEL_DIR / "metadata.json") as f:
        meta = json.load(f)
    DECISION_THRESHOLD = meta["decision_threshold"]

    print("=== Estrategia de reentrenamiento ===\n")
    for name, X_batch, y_batch in [
        ("BATCH_1 (sin drift)", BATCH_1_X, BATCH_1_y),
        ("BATCH_2 (drift moderado)", BATCH_2_X, BATCH_2_y),
        ("BATCH_3 (drift fuerte)", BATCH_3_X, BATCH_3_y),
    ]:
        psi_report = compute_drift_report(REFERENCE, X_batch, numeric_cols, categorical_cols)
        y_proba = modelo.predict_proba(X_batch)[:, 1]
        y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
        metrics = compute_model_metrics(y_batch, y_pred, y_proba)

        decision = decide_retraining(psi_report, metrics["pr_auc"])
        decision["batch"] = name
        decision["timestamp"] = datetime.now(timezone.utc).isoformat()

        print(f"--- {name} ---")
        print(f"  Features en alerta PSI: {decision['features_en_alerta']}")
        print(f"  PR-AUC: {decision['pr_auc']}")
        print(f"  Trigger retraining: {decision['trigger_retraining']}")
        print(f"  Razón: {decision['razon']}\n")

        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(decision) + "\n")
