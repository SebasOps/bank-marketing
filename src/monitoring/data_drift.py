"""
Data Monitoring (O2): compara P_reference(X) vs P_production(X) con PSI.

REFERENCE       = X_train (distribución con la que se entrenó el modelo)
PRODUCTION_BATCH = particiones de X_test_final (datos no vistos), simulando
                    3 lotes consecutivos de producción, con drift progresivo
                    inyectado en BATCH_2 y BATCH_3 para demostrar detección.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.pipelines.split import split_data
from src.features.build_features import feature_selection, encode_target
from src.tracking.config import RANDOM_SEED

# --------------------------------------------
# PSI

def calculate_psi_numeric(reference, production, bins=10, epsilon=1e-4):
    breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    prod_counts, _ = np.histogram(production, bins=breakpoints)

    ref_perc = np.where(ref_counts == 0, epsilon, ref_counts / len(reference))
    prod_perc = np.where(prod_counts == 0, epsilon, prod_counts / len(production))

    return float(np.sum((prod_perc - ref_perc) * np.log(prod_perc / ref_perc)))


def calculate_psi_categorical(reference, production, epsilon=1e-4):
    categories = set(reference.unique()) | set(production.unique())
    ref_perc = reference.value_counts(normalize=True)
    prod_perc = production.value_counts(normalize=True)

    psi = 0.0
    for cat in categories:
        r = max(ref_perc.get(cat, 0), epsilon)
        p = max(prod_perc.get(cat, 0), epsilon)
        psi += (p - r) * np.log(p / r)
    return float(psi)


def interpret_psi(psi):
    if psi < 0.1:
        return "OK"
    elif psi < 0.25:
        return "WARNING"
    return "ALERT"


def compute_drift_report(reference_df, production_df, numeric_cols, categorical_cols):
    report = {}
    for col in numeric_cols:
        psi = calculate_psi_numeric(reference_df[col], production_df[col])
        report[col] = {"psi": round(psi, 4), "status": interpret_psi(psi)}
    for col in categorical_cols:
        psi = calculate_psi_categorical(reference_df[col], production_df[col])
        report[col] = {"psi": round(psi, 4), "status": interpret_psi(psi)}
    return report


# --------------------------------------------
# Inyección de drift sintético (solo en memoria, nunca toca el CSV original)

def inject_drift(df, numeric_col="balance", categorical_col="job", intensity=0.0):
    df_drifted = df.copy()
    if intensity > 0:
        shift = df_drifted[numeric_col].std() * intensity * 2
        df_drifted[numeric_col] = df_drifted[numeric_col] + shift

        n_affected = int(len(df_drifted) * intensity * 0.5)
        idx = df_drifted.sample(n=n_affected, random_state=RANDOM_SEED).index
        df_drifted.loc[idx, categorical_col] = "retired"
    return df_drifted


# --------------------------------------------
# Main

if __name__ == "__main__":
    PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "bank_marketing.csv"
    df = pd.read_csv(PROCESSED_PATH)

    X = feature_selection(df)
    y = encode_target(df["y"])
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Misma partición que threshold_analysis.py
    _, X_test_final, _, _ = train_test_split(
        X_test, y_test, test_size=0.5, stratify=y_test, random_state=RANDOM_SEED
    )

    REFERENCE = X_train.reset_index(drop=True)
    batches = np.array_split(X_test_final.reset_index(drop=True), 3)
    BATCH_1 = batches[0]
    BATCH_2 = inject_drift(batches[1], intensity=0.5)
    BATCH_3 = inject_drift(batches[2], intensity=1.0)

    numeric_cols = REFERENCE.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = REFERENCE.select_dtypes(exclude=[np.number]).columns.tolist()

    for name, batch in [("BATCH_1 (sin drift)", BATCH_1),
                         ("BATCH_2 (drift moderado)", BATCH_2),
                         ("BATCH_3 (drift fuerte)", BATCH_3)]:
        print(f"\n=== {name} ===")
        report = compute_drift_report(REFERENCE, batch, numeric_cols, categorical_cols)
        for feature, r in report.items():
            print(f"  {feature}: PSI={r['psi']} -> {r['status']}")
