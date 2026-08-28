"""
Split de datos para el dataset Bank Marketing.

Contiene la lógica de división entre entrenamiento y prueba, exclusiva de
la etapa de entrenamiento (no se utiliza en producción, donde no existe
un conjunto de "prueba" sino datos nuevos a predecir). Se centraliza para
que notebooks y scripts de entrenamiento (train.py) usen siempre el mismo
test_size, random_seed y estrategia de stratify, evitando splits inconsistentes
entre distintas partes del proyecto.
"""

from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.20, random_seed=42):
    """
    Divide el dataset en entrenamiento y prueba, manteniendo la proporción
    de clases (stratify) dado el desbalance identificado en el EDA.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y
    )
