# Bank Marketing

---

## Business Problem

### Objetivo 
El objetivo que se presenta para este caso son los siguietes: 
* Identifizar clientes con mayor probabilidad de contratar un depósito a plazo

Para lograrlo el desarrollo se enfocará en encontrar la configuración correcta para maximizar el acurracy del modelo de clasificación.

---
## Dataset

Los datos del dataset están relacionados con campañas de marketing directo de una institución bancaria portuguesa. Las campañas de marketing se basaban en llamadas telefónicas.

* Fuente: UC Irvine - Maching Learning Repository
* Características: 16
* Instancias: 45211
* Tipos de caraterísticas: Categóricas y Enteros
* Área: Negocios

### Variables del dataset

| Variable | Rol | Tipo | Demografía | Descripción | Unidades | Valores Faltantes |
|----------|------|------|-------------|-------------|-----------|-------------------|
| age | Feature | Entero | Edad | Edad del cliente. | | No |
| job | Feature | Categórico | Ocupación | Tipo de trabajo ('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'). | | No |
| marital | Feature | Categórico | Estado civil | Estado civil ('divorced','married','single','unknown'). Nota: 'divorced' incluye divorciados y viudos. | | No |
| education | Feature | Categórico | Nivel educativo | Nivel educativo ('basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown'). | | No |
| default | Feature | Binario | | ¿Tiene créditos en incumplimiento de pago? | | No |
| balance | Feature | Entero | | Saldo promedio anual. | Euros | No |
| housing | Feature | Binario | | ¿Tiene préstamo hipotecario? | | No |
| loan | Feature | Binario | | ¿Tiene préstamo personal? | | No |
| contact | Feature | Categórico | | Tipo de comunicación utilizada para contactar al cliente ('cellular','telephone'). | | Sí |
| day_of_week | Feature | Fecha | | Último día de la semana en que se contactó al cliente. | | No |
| month | Feature | Fecha | | Último mes del año en que se contactó al cliente ('jan', 'feb', 'mar', ..., 'nov', 'dec'). | | No |
| duration | Feature | Entero | | Duración del último contacto, en segundos. Nota importante: este atributo influye considerablemente en la variable objetivo (p. ej., si la duración es 0, entonces y='no'). Sin embargo, la duración se desconoce antes de realizar la llamada. Además, una vez finalizada la llamada, el valor de y ya se conoce. Por tanto, esta variable de entrada solo debe incluirse con fines de referencia y ha de descartarse si se pretende obtener un modelo predictivo realista. | | No |
| campaign | Feature | Entero | | Número de contactos realizados durante esta campaña para este cliente (incluyendo el último contacto). | | No |
| pdays | Feature | Entero | | Número de días transcurridos desde que el cliente fue contactado por una campaña anterior (-1 significa que nunca fue contactado previamente). | | Sí |
| previous | Feature | Entero | | Número de contactos realizados antes de esta campaña para este cliente. | | No |
| poutcome | Feature | Categórico | | Resultado de la campaña de marketing anterior ('failure','nonexistent','success'). | | Sí |
| y | Target | Binario | | ¿El cliente se suscribió a un depósito a plazo fijo? | | No |

---

## Architecture

---

## Repository Structure

### Estructura 
El repositorio en Github del proyecto tiene la siguiente estructura:

bank-marketing/
├── src/
│   ├── data-quality/
|   |   ├── clean.py
|   |   └── gates.py
│   └── ingestion/
|       └── ingest.py
|
├── notebooks/
|   ├── data-quality.ipynb
|   └── data-eda.ipynb
|
├── tests/
|   └── test_clean_gates.py
|
├── .gitignore
├── requirements.txt
└── README.md

### Ramas
Para el desarrollo se usaron las siguientes ramás: 
* main
* develop
* feature/data-cleaning
* ...

---

## Installation

### Programas necesarios
Para ejecutar este proyecto de MLOps se debe contar con los programas que serán mencionados, la documentación de este proyecto partirá de que se tienen ya instalados y configurados, pero igualmente se adjuntan los links de instalación, mas no serán explicados: 
* Python - https://www.python.org/downloads/ 
* Git - https://git-scm.com/install/ 
* Visual Studio Code - https://code.visualstudio.com/download?_exp_download=fb315fc982

### Clonar repositorio
Seguir los siguientes pasos para obtener el proyecto desde el repositorio de Github: 
1. Abrir una nueva terminal en Visual Studio Code (o el editor de código de preferencia).
2. Escribir y correr el siguiente comando: git clone https://github.com/SebasOps/bank-marketing

### Entorno virtual ??

---

## Data Ingestion

El dataset (Bank Marketing, UCI ML Repository, id=222) se obtiene mediante:

​```bash
python src/ingestion/ingest.py
​```

Esto descarga los datos directamente desde UCI usando la librería `ucimlrepo` y los guarda en `data/raw/bank_marketing.csv`. No se requiere ningún archivo local previo. Script es completamente reproducible.

### Requisitos
- Instalar dependencias: `pip install -r requirements.txt`

---

## Training

### Mediciones a evaluar: 
* PR-AUC: será el criterio principal para la selección entre modelos/configuraciones. Esta métrica indica que tan bien el modelo distinge "yes" en general, además es ideal para evitar un número "bonito" que pueden dar otras métricas como accuracy por el desbalanceo de las clases.
* Recall: segunda prioridad. En una campaña de marketing, no detectar a alguien que sí se iba a suscribir (falso negativo) suele costar más que llamar de más a alguien que no se suscribe (falso positivo), y esto es lo que nos refleja la métrica, de los "yes" reales, cual porcentaje detectó el modelo.
* F1: se utilizará para revisar que un posible recall alto no se deba por una precisión baja, por ende, un F1 bajo y un recall alto signifiaría que el recall se debe a que el modelo simplemente predice "yes" para todo.
* Accuracy: se identificó que no es la métrica ideal por el desbalanceo de las clases, pero igualmente se reportará. 


### Hiperparámetros utilizados en los experimentos

Las siguientes constantes se declararon en src/tracking/config.py para importarlas a src/training.py y src/evaluation/threshold_analysis.py evitando duplicidad

RANDOM_SEED = 42
CLASS_BALANCED = "balanced"

INCLUIR_SMOTE_KNN = True
NO_INCLUIR_SMOTE_KNN = False

INCLUIR_ESCALADO = True
NO_INCLUIR_ESCALADO = False

Todos los modelos utilizan RANDOM_SEED para conservar los mismos resultados sin importar quien ejecute los experimentos asegurando reproducibilidad.


### Modelos/configuraciones probados

#### Random Forest
* Experimento 1: max_depth=5, n_estimator=100, class_weight="balanced", incluir_escalado=NO_INCLUIR_ESCALADO
* Experimento 2: max_depth=10, n_estimator=100, class_weight="balanced", incluir_escalado=NO_INCLUIR_ESCALADO 

Se observa que el experimento 2 presentó resultados mejores en las métricas:

* Experimento 1: 
* Experimento 2:

#### Decision Tree

* Experimento 3: max_depth=10, class_weight="balanced", incluir_escalado=NO_INCLUIR_ESCALADO
* Experimento 4: max_depth=15, class_weight="balanced", incluir_escalado=NO_INCLUIR_ESCALADO

Se observó que el experimento 3 presentó mejores resultados en las métricas entre los experimentos de Decision Tree: 

* Experimento 3: 
* Experimento 4:

#### KNN

* Experimento 5: 
* Experimento 6:
* Experimento 7:

#### Logistic Regression

* Experimento 8: 
* Experimento 9:

* RF 10 depth mejor que 5 
* DT 10 depth mejor que 15
* KNN
* RL

Ganó el RF de 10 depth
Queda ir modificando el random forest- 15? - n_estimators

---

## MLflow

---

## Docker

---

## API

---

## Monitoring

---

## Results

---

## Team

El equipo se conforma por:

* Sebastián Aguilar Benavides
* María Paula Elizondo Herrera

Donde cada uno de los integrantes aportó todas y cada una de las diferentes secciones.

---

## Comandos reproducibles 

* pip install -r requirements.txt
* python src/ingestion/ingest.py
* python src/data-quality/clean.py
* python src/data-quality/gates.py
* python src/tests/test_clean_gates.py
* python src/training.py
* python src/evalution/threshold_analysis.py

---


