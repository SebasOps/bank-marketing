"""
Función centralizada de tracking MLflow.

Encapsula el ciclo completo de un experimento (fit, predicción, cálculo de
métricas, y logging de parámetros/métricas/artifacts) en un único punto
reutilizable por los cuatro modelos del proyecto, evitando duplicar la
lógica para tracking MLflow.

Soporta dos modos de uso:
- Entrenamiento normal (already_fitted=False): entrena el pipeline recibido
  y lo registra como artifact en el run.
- Evaluación sobre un modelo ya entrenado (already_fitted=True): no
  reentrena; se usa para análisis posteriores al modelo ganador, como la
  evaluación con un umbral de decisión distinto al default.
"""

# Imports
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score,
    accuracy_score, ConfusionMatrixDisplay,
)


def run_experiment(
    pipeline, model_name, params, X_train, y_train, X_test, y_test, run_name,
    already_fitted=False, threshold=0.5
):
    with mlflow.start_run(run_name=run_name):
        if not already_fitted:
            pipeline.fit(X_train, y_train)

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            "f1": f1_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "acc_score": accuracy_score(y_test, y_pred),
        }

        mlflow.log_param("algorithm", model_name)
        mlflow.log_param("decision_threshold", threshold)
        for k, v in params.items():
            mlflow.log_param(k, v)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        if not already_fitted:
            mlflow.sklearn.log_model(pipeline, "model")

        print(f"[{run_name}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        return metrics
