"""
Constantes centralizadas para los experimientos 
de tracking con MLflow. Se importan evitando 
duplicidad en lo que sea posible e ideal.
"""

RANDOM_SEED = 42
CLASS_BALANCED = "balanced"

INCLUIR_SMOTE_KNN = True
NO_INCLUIR_SMOTE_KNN = False

INCLUIR_ESCALADO = True
NO_INCLUIR_ESCALADO = False

def get_data_version(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]
