from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo


# Ruta raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Carpeta donde almacenaremos los datos raw
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def ingest_data():
    """Descarga el dataset Bank Marketing desde UCI
    y lo guarda en formato CSV dentro de data/raw.
    """

    # Crear la carpeta si no existe
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Obteniendo Bank Marketing desde UCI...")

    # Descargar dataset desde UCI
    bank_marketing = fetch_ucirepo(id=222)

    # Obtener los datos
    X = bank_marketing.data.features
    y = bank_marketing.data.targets

    # Unir variables predictoras y variable objetivo
    df = pd.concat([X, y], axis=1)

    # Ruta de salida
    output_path = RAW_DIR / "bank_marketing.csv"

    # Guardar dataset
    df.to_csv(output_path, index=False)

    print("Ingesta completada correctamente.")
    print(f"Dataset guardado en: {output_path}")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")


if __name__ == "__main__":
    ingest_data()