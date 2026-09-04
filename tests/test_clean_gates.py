# Imports
import sys
import pandas as pd
from pathlib import Path


# Ruta raíz del proyecto (cwd = donde se encuentra el notebook; .parents[1] = ruta padre = ruta raíz)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "bank_marketing.csv"


# Agrega la raíz al sistema de rutas de Python si no está ya presente
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# Luego de obtener la ruta raíz
from src.quality.clean import clean_data
from src.quality.gates import data_quality_gates


# Pipeline
def test_clean_and_gates_run_without_errors():

    df_raw = pd.read_csv(RAW_PATH)

    # Pasar por limpieza
    df_limpio = clean_data(df_raw)

    # Pasar por gates
    df_validado = data_quality_gates(df_limpio)
    
