# Imports
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score,
    accuracy_score, ConfusionMatrixDisplay,
)


def run_experiment(pipeline, model_name, params, X_train, y_train, X_test, y_test, run_name):
    with mlflow.start_run(run_name=run_name):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "f1": f1_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "acc_score": accuracy_score(y_test, y_pred),
        }

        mlflow.log_param("algorithm", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        mlflow.sklearn.log_model(
            pipeline,
            "model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )

        print(f"[{run_name}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        return metrics