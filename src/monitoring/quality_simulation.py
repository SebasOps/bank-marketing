"""
Simulación obligatoria de problemas de calidad (Q).

Contamina una COPIA en memoria de un batch de producción con los 6 tipos de
defectos requeridos, y verifica que production_quality_gates() los detecte,
bloquee (AssertionError) y registre el incidente. El dataset original nunca
se toca: toda la contaminación ocurre sobre copias descartables.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.pipelines.split import split_data
from src.features.build_features import feature_selection, encode_target
from src.quality.clean import resolve_types
from src.quality.gates import production_quality_gates
from src.tracking.config import RANDOM_SEED


LOG_PATH = PROJECT_ROOT / "logs" / "quality_incidents.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)


# --------------------------------------------
# Funciones de contaminación (todas devuelven una copia, nunca mutan el original)

def contaminate_missing_values(df, col="balance", frac=0.1):
    df_c = df.copy()
    idx = df_c.sample(frac=frac, random_state=RANDOM_SEED).index
    df_c.loc[idx, col] = np.nan
    return df_c


def contaminate_duplicates(df, n=5):
    df_c = df.copy()
    filas_duplicadas = df_c.sample(n=n, random_state=RANDOM_SEED)
    return pd.concat([df_c, filas_duplicadas], ignore_index=True)


def contaminate_extreme_outlier(df, col="balance", value=-500000, n=3):
    df_c = df.copy()
    idx = df_c.sample(n=n, random_state=RANDOM_SEED).index
    df_c.loc[idx, col] = value
    return df_c


def contaminate_incorrect_datatype(df, col="age", n=3):
    df_c = df.copy()
    df_c[col] = df_c[col].astype(object)
    idx = df_c.sample(n=n, random_state=RANDOM_SEED).index
    df_c.loc[idx, col] = "treinta"
    return df_c


def contaminate_unknown_category(df, col="job", value="UNKNOWN_NEW_CATEGORY", n=3):
    df_c = df.copy()
    idx = df_c.sample(n=n, random_state=RANDOM_SEED).index
    df_c.loc[idx, col] = value
    return df_c


def contaminate_schema_modification(df):
    df_c = df.copy()
    df_c["columna_inesperada"] = 1
    df_c = df_c.drop(columns=[df_c.columns[0]])
    return df_c


# --------------------------------------------
# Runner: aplica gates, captura resultado, registra incidente

def run_gate_check(scenario_name, contaminated_df, reference_df, gate_fn, needs_reference=True, needs_type_coercion=False):
    """
    Corre UN gate específico (no toda la cadena de production_quality_gates),
    para que cada escenario de contaminación pruebe exclusivamente el gate
    diseñado para detectarlo, sin que otro gate lo intercepte antes por
    coincidencia. needs_reference indica explícitamente si el gate requiere
    reference_df como segundo argumento (evita adivinar la firma por
    introspección, que es frágil ante parámetros con default).
    """
    df_to_check = resolve_types(contaminated_df) if needs_type_coercion else contaminated_df

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario_name,
        "n_rows": len(df_to_check),
    }

    try:
        if needs_reference:
            gate_fn(df_to_check, reference_df)
        else:
            gate_fn(df_to_check)
        entry["detected"] = False
        entry["blocked"] = False
        entry["message"] = "El gate correspondiente no detectó el defecto"
        print(f"[FALLO DEL GATE] {scenario_name}: no se detectó el defecto")
    except AssertionError as e:
        entry["detected"] = True
        entry["blocked"] = True
        entry["message"] = str(e)
        print(f"[OK] {scenario_name}: detectado y bloqueado -> {e}")

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


if __name__ == "__main__":
    PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "bank_marketing.csv"
    df = pd.read_csv(PROCESSED_PATH)

    X = feature_selection(df)
    y = encode_target(df["y"])
    X_train, X_test, y_train, y_test = split_data(X, y)

    REFERENCE = X_train.reset_index(drop=True)
    BATCH_LIMPIO = X_test.reset_index(drop=True).head(200)  # batch pequeño representativo, sin contaminar

    # Validación de baseline: el batch limpio, sin contaminar, DEBE pasar los
    # gates. Si falla acá, cualquier resultado "detectado" en los escenarios
    # de abajo es sospechoso (no se sabe si detectó la contaminación o ruido
    # preexistente del batch).
    try:
        production_quality_gates(BATCH_LIMPIO, REFERENCE)
        print("[OK] Baseline (batch limpio) pasa los gates sin alertas falsas.\n")
    except AssertionError as e:
        print(f"[ADVERTENCIA] El batch limpio NO pasa los gates: {e}")
        print("Los resultados de los escenarios de abajo pueden no ser confiables.\n")

    from src.quality.gates import (
        check_no_missing_values, check_no_duplicates,
        check_no_extreme_outliers, check_valid_categories,
        expected_feature_columns,
    )

    # (nombre, df_contaminado, gate_fn, necesita_reference_df, necesita_coercion_tipos)
    escenarios = [
        ("missing_values", contaminate_missing_values(BATCH_LIMPIO), check_no_missing_values, False, False),
        ("duplicated_rows", contaminate_duplicates(BATCH_LIMPIO), check_no_duplicates, False, False),
        ("extreme_outlier", contaminate_extreme_outlier(BATCH_LIMPIO), check_no_extreme_outliers, True, False),
        ("incorrect_datatype", contaminate_incorrect_datatype(BATCH_LIMPIO), check_no_missing_values, False, True),
        ("unknown_category", contaminate_unknown_category(BATCH_LIMPIO), check_valid_categories, True, False),
        ("schema_modification", contaminate_schema_modification(BATCH_LIMPIO), expected_feature_columns, True, False),
    ]

    print("=== Simulación de contaminación de calidad ===\n")
    resultados = []
    for nombre, df_contaminado, gate_fn, necesita_ref, necesita_coercion in escenarios:
        resultado = run_gate_check(nombre, df_contaminado, REFERENCE, gate_fn, necesita_ref, necesita_coercion)
        resultados.append(resultado)

    n_detectados = sum(r["detected"] for r in resultados)
    print(f"\n{n_detectados}/{len(resultados)} escenarios detectados y bloqueados correctamente.")
