"""
Exporta un modelo desde el MLflow Model Registry a una carpeta local
autocontenida, lista para copiar dentro de la imagen Docker.

Correr UNA VEZ, en el entorno local, con el mlflow server ya levantado
(mismo tracking URI que usaste para entrenar/registrar el modelo).

Uso:
    python export_model.py
"""

import json
import shutil
from pathlib import Path
import mlflow
import mlflow.sklearn
from mlflow.sklearn import SERIALIZATION_FORMAT_CLOUDPICKLE

# ---------------------------------------------

TRACKING_URI = "http://127.0.0.1:5000"
MODEL_URI = "models:/bank-marketing-model@production"
OUTPUT_DIR = "model_artifact"

# ---------------------------------------------

mlflow.set_tracking_uri(TRACKING_URI)

output_path = Path(OUTPUT_DIR)
if output_path.exists():
    shutil.rmtree(output_path)  # save_model falla si la carpeta ya existe

print(f"Descargando modelo desde: {MODEL_URI}")
modelo = mlflow.sklearn.load_model(MODEL_URI)

print(f"Guardando modelo autocontenido en: {OUTPUT_DIR}/")
mlflow.sklearn.save_model(
    modelo,
    OUTPUT_DIR,
    serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
)


# ---------------------------------------------
# Metadata que la API necesita y que no viaja dentro del pipeline sklearn
# ---------------------------------------------

client = mlflow.MlflowClient()
mv = client.get_model_version_by_alias("bank-marketing-model", "production")

metadata = {
    "model_version": mv.version,
    "decision_threshold": float(mv.tags["decision_threshold"]),
}

with open(output_path / "metadata.json", "w") as f:
    json.dump(metadata, f)

print(f"Metadata guardada: {metadata}")

# ---------------------------------------------

print("Listo. Copiar esta carpeta dentro del proyecto Docker (junto al Dockerfile).")
print("El contenedor ya NO necesitará conectarse al mlflow server para predecir.")
