# Bank Marketing

---



## Business Problem

### Contexto
Una institución financiera realiza campañas telefónicas para vender depósitos a plazo fijo.


### Objetivo
El objetivo no es únicamente maximizar el accuracy del modelo de clasificación, sino identificar correctamente a los clientes con mayor probabilidad de conversión (contratar el depósito a plazo). Dado el desbalanceo de clases del dataset (~88% "no" / 12% "yes"), maximizar accuracy de forma aislada favorecería un modelo que predice "no" casi siempre, sin utilidad real para la campaña. Por ello, el desarrollo se enfoca en encontrar la configuración que mejor distinga a los clientes con intención de conversión, priorizando PR-AUC y recall sobre accuracy (ver sección *Training* para el detalle de la justificación de métricas).



---



## Dataset
Los datos del dataset están relacionados con campañas de marketing directo de una institución bancaria portuguesa. Las campañas de marketing se basaban en llamadas telefónicas.

* Fuente: UC Irvine - Machine Learning Repository
* Características: 16 (más la variable objetivo `y`)
* Instancias: 45211
* Tipos de características: Categóricas, Binarias, Enteras y de Fecha
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
El repositorio en Github de este proyecto tiene la siguiente estructura:

```python
bank-marketing/
├── app/
│   └── main.py
├── interface/
│   └── app.py
├── model_artifact/                ??? 
│   ├── conda.yaml
│   ├── metadata.json
│   ├── MLmodel
│   ├── model.pkl
│   ├── python_env.yaml
│   └── requirements.txt
├── notebooks/
│   ├── data-eda.ipynb
│   ├── data-quality.ipynb
│   └── feature-engineering.ipynb
├── src/
│   ├── evaluation/
│   │   └── threshold_analysis.py
│   ├── features/
│   │   └── build_features.py
│   ├── ingestion/
│   │   └── ingest.py
│   ├── pipelines/
│   │   └── split.py
│   ├── quality/
│   │   ├── clean.py
│   │   └── gates.py
│   ├── tracking/
│   │   ├── config.py
│   │   └── run_experiment.py
│   └── training.py
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_model.py
│   └── test_clean_gates.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── export_model.py
├── bank.db
├── requirements-api.txt
├── requirements-dev.txt ?? 
├── requirements.txt
└── README.md
```


### Ramas
Para el desarrollo se usaron las siguientes ramas:
* **main**: rama de producción, únicamente se sube acá si en la rama 'develop' se confirma funcionalidad completa.
* **develop**: rama para probar todo en conjunto.
* **feature/data-cleaning**: se realizó el EDA, la limpieza y gates.
* **feature/model**: experimentos, modelo ganador, análisis de umbrales.
* **feature/api**: construcción de app.py (API), creación de la imagen y contenedor en Docker y pruebas de datos, modelo y API.



---



## Installation

### Programas necesarios
Para ejecutar este proyecto de MLOps se debe contar con los programas que serán mencionados. La documentación de este proyecto parte de que ya están instalados y configurados y que se realiza en Windows, pero igualmente se adjuntan los links de instalación, más no serán explicados:
* Python - https://www.python.org/downloads/
* Git - https://git-scm.com/downloads
* Docker - https://www.docker.com/products/docker-desktop/
* Visual Studio Code - https://code.visualstudio.com/download


### Clonar repositorio
Seguir los siguientes pasos para obtener el proyecto desde el repositorio de Github:
1. Abrir una nueva terminal en Visual Studio Code (o el editor de código de preferencia).
2. Escribir y correr el siguiente comando: `git clone https://github.com/SebasOps/bank-marketing`


### Entorno virtual
1. Crear el entorno virtual ejecutando el siguiente comando: ```python -m venv mlflow-project```
2. Activar el entorno virtual: ```mlflow-project/Scripts/activate```
3. Instalar dependencias ejecutando el siguiente comando: ```pip install -r requirements.txt```



---



## Data Ingestion

El dataset se obtiene mediante la ejecución del siguiente comando: ```python src/ingestion/ingest.py```

Esto descarga los datos directamente desde UCI usando la librería `ucimlrepo` y los guarda en `data/raw/bank_marketing.csv`. No se requiere ningún archivo local previo ni conexión a bases de datos externas más allá de UCI; el script es completamente reproducible.

### Requisitos
- Entorno virtual activo con las dependencias instaladas (ver sección *Installation*).



---



## Training

### Mediciones a evaluar:
* **PR-AUC**: será el criterio principal para la selección entre modelos/configuraciones. Esta métrica indica que tan bien el modelo distinge "yes" en general, además es ideal para evitar un número "bonito" que pueden dar otras métricas como accuracy por el desbalanceo de las clases.
* **Recall**: segunda prioridad. En una campaña de marketing, no detectar a alguien que sí se iba a suscribir (falso negativo) suele costar más que llamar de más a alguien que no se suscribe (falso positivo), y esto es lo que nos refleja la métrica, de los "yes" reales, cual porcentaje detectó el modelo.
* **F1**: se utilizará para revisar que un posible recall alto no se deba por una precisión baja, por ende, un F1 bajo y un recall alto significaría que el recall se debe a que el modelo simplemente predice "yes" para todo.
* **Accuracy**: se identificó que no es la métrica ideal por el desbalanceo de las clases, pero igualmente se reportará.


### Hiperparámetros utilizados en los experimentos

Las siguientes constantes se declararon en ```src/tracking/config.py``` para importarlas a ```src/training.py``` y ```src/evaluation/threshold_analysis.py``` evitando duplicidad:

```python
RANDOM_SEED = 42
CLASS_BALANCED = "balanced"

INCLUIR_SMOTE_KNN = True
NO_INCLUIR_SMOTE_KNN = False

INCLUIR_ESCALADO = True
NO_INCLUIR_ESCALADO = False
```

Todos los experimentos utilizan ```RANDOM_SEED``` para conservar los mismos resultados sin importar quien ejecute los experimentos asegurando reproducibilidad y consistencia.


### Modelos/configuraciones probados

#### Random Forest
* Experimento 1: max_depth=5, n_estimator=100, class_weight="balanced", incluir_escalado=NO_INCLUIR_ESCALADO
* Experimento 2: max_depth=10, n_estimator=100, class_weight="balanced", incluir_escalado=NO_INCLUIR_ESCALADO

#### Decision Tree

* Experimento 3: max_depth=10, class_weight="balanced", incluir_escalado=NO_INCLUIR_ESCALADO
* Experimento 4: max_depth=15, class_weight="balanced", incluir_escalado=NO_INCLUIR_ESCALADO

#### KNN

* Experimento 5: n_neighbors=15, incluir_escalado=INCLUIR_ESCALADO, incluir_smote=INCLUIR_SMOTE_KNN, random_seed=RANDOM_SEED
* Experimento 6: n_neighbors=31, incluir_escalado=INCLUIR_ESCALADO, incluir_smote=INCLUIR_SMOTE_KNN, random_seed=RANDOM_SEED

Se observó que entre el experimento 5 y el 6, este segundo presentó mejores resultados, por ende, se realizó el mismo ejercicio, pero sin aplicarle la técnica SMOTE de balanceo.

* Experimento 7: n_neighbors=31, incluir_escalado=INCLUIR_ESCALADO, incluir_smote=NO_INCLUIR_SMOTE_KNN, random_seed=RANDOM_SEED

#### Logistic Regression

* Experimento 8: C=0.4, class_weight=CLASS_BALANCED, max_iter=1000, incluir_escalado=INCLUIR_ESCALADO
* Experimento 9: C=0.6, class_weight=CLASS_BALANCED, max_iter=1000, incluir_escalado=INCLUIR_ESCALADO


### Resultados obtenidos
De los primeros 9 experimentos, inicialmente se hizo una comparación entre los experimentos del mismo modelo. Se observó que el experimento 2 fue el que mejores resultados presentó: 

* PR-AUC = 0.43
* Recall = 0.62
* F1 = 0.43
* Accuracy = 0.81

Aunque fue el mejor entre los experimentos realizados no significó que fuera excelente. Para ver si el modelo mejoraba, se realizaron más experimentos basados en este candidato.


### Experimentos en base al candidato

Primero, se varió el hiperparámetros n_estimator, corriendo dos nuevos ejercicios, uno con 200 y otro con 300. El valor de n_estimator es la cantidad de árboles distintos que el modelo de random forest realiza para obtener sus clasificaciones:

* Experimento 10: max_depth=5, n_estimator=200, class_weight="balanced", incluir_escalado=NO_INCLUIR_ESCALADO
* Experimento 11: max_depth=5, n_estimator=300, class_weight="balanced", incluir_escalado=NO_INCLUIR_ESCALADO

Realmente no hubo mejora sustancial en comparación al modelo candidato, por ende, se probó modificando hiperparámetro, esta vez fue min_samples_leaf, se realizaron los siguientes experimentos:

* Experimento 12: max_depth=10, n_estimators=100, min_samples_leaf=5, class_weight=CLASS_BALANCED, incluir_escalado=NO_INCLUIR_ESCALADO
* Experimento 13: max_depth=10, n_estimators=100, min_samples_leaf=10, class_weight=CLASS_BALANCED, incluir_escalado=NO_INCLUIR_ESCALADO
* Experimento 14: max_depth=10, n_estimators=100, min_samples_leaf=15, class_weight=CLASS_BALANCED, incluir_escalado=NO_INCLUIR_ESCALADO

Con estos cambios en el hiperparámetro min_samples_leaf hubo una regresión con respecto al modelo candidato base, el cual usa min_samples_leaf=1 por defecto. Por ese motivo se descartó dichos cambios y se mantiene el modelo candidato sin modificación. 

A este punto se decidió no modificar más hiperparámetros, unicamente se haría un análisis de umbral basándonos en que el recall nos demostraba que el modelo aun no capturaba un gran porcentaje de los registros "yes" reales (sí contrataron el depósito); se realizarón los siguientes experimentos:

* Experimento 15: max_depth=10, n_estimators=100, class_weight=CLASS_BALANCED, incluir_escalado=NO_INCLUIR_ESCALADO, threshold=0.45
* Experimento 16: max_depth=10, n_estimators=100, class_weight=CLASS_BALANCED, incluir_escalado=NO_INCLUIR_ESCALADO, threshold=0.40

Se observó que ambos experimentos aumentaban el recall a costa de la precisión y el F1 respecto al modelo candidato base. El PR-AUC también fue menor que en el candidato base (0.435 vs ~0.41), pero esta diferencia no puede atribuirse al cambio de umbral, ya que esta métrica es invariante al punto de corte (se calcula sobre las probabilidades predichas, no sobre la clase discreta resultante). Se confirma esto comparando los Experimentos 15 y 16 entre sí: ambos usan el mismo conjunto de test y distinto umbral (0.45 y 0.40), y su PR-AUC es prácticamente idéntico (0.41). La diferencia de PR-AUC frente al candidato base se explica, en cambio, porque este último se evaluó sobre `X_test` completo, mientras que los Experimentos 15 y 16 se evaluaron sobre `X_test_thr`, la mitad reservada para el análisis de umbral.

* Experimento 17: max_depth=10, n_estimators=100, class_weight=CLASS_BALANCED, incluir_escalado=NO_INCLUIR_ESCALADO, threshold=0.45

Este último experimento contiene los mismos hiperparámetros que el *Experimento 15*, pero observamos que mejoraba en comparación al mismo. La comparación final del modelo base candidato y el *Experimento 17* el cual tiene el umbral de 0.45 es la siguiente

| | Candidato base | Modelo umbral 0.45 |
|-|----------------|--------------------|
| PR-AUC | 0.43 | 0.46 |
| Recall | 0.62 | 0.72 |
| F1 | 0.43 | 0.40 |
| Accuracy | 0.81 | 0.75 |

Cabe aclarar que esta comparación no es sobre el mismo conjunto de datos: el candidato base (Experimento 2) fue evaluado sobre el conjunto de test completo, mientras que el Experimento 17 fue evaluado únicamente sobre `X_test_final`, la mitad del test original reservada como holdout no vista durante la selección del umbral (la otra mitad, `X_test_thr`, se usó para los Experimentos 15 y 16). Esta partición se hizo intencionalmente para evitar que el umbral se eligiera y evaluara sobre los mismos datos, pero implica que el Experimento 17 se midió sobre una muestra de menor tamaño que el candidato base.

Viendo sus métricas se observa que, PR-AUC fue mayor con el umbral de 0.45 (aclarado anteriormente que se debe a que usan distintos datos de test), esto se puede deber a que como capturó mayor cantidad de "yes" reales, logró identificar mejor dicha clase; se ve que el recall igualmente aumentó, por ende, logró capturar el mayor porcentaje de los "yes" reales; por otro lado, tanto F1 como accuracy bajaron, esto debido a la baja en la precisión del modelo.

La decisión fue tomada en base al objetivo: "identificar correctamente clientes con mayor probabilidad de conversión". Conocíamos claramente la solicitud de "maximizar accuracy", pero al identificar el gran desbalance de las clases se entendió que podría no ser la métrica que mejor refleje el comportamiento del modelo.



---



## MLflow

### Levantar UI
Activar y acceder a la interfaz de MLflow:
1. Con el venv activo, ejecutar el siguiente comando:
````mlflow server --backend-store-uri sqlite:///bank.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000```
2. En el navegador, acceder a: http://127.0.0.1:5000


### Correr experimentos
Ejecutar el siguiente comando en la terminal: ```python src/training.py```

Esto corre los 13 experimentos de comparación inicial entre los cuatro modelos (Random Forest, Decision Tree, KNN y Logistic Regression, incluyendo la búsqueda de `min_samples_leaf` para Random Forest, ver a detalle en la sección *Training*). Los experimentos se podrán visualizar en el panel de MLflow. Sigue esta ruta: Experiments > classification-bank-marketing > Training runs.

El análisis de umbral (Experimentos 15-17) se corre por separado. Corre el siguiente comando en la terminal: ```python src/evalution/threshold_analysis.py```


### Registrar el modelo ganador
El modelo ganador (Random Forest, `max_depth=10`, `n_estimators=100`) se registra manualmente en el Model Registry desde la UI de MLflow, siguiendo estos pasos:

1. **Registrar como *candidate***: en la UI, ubicar el run `rf-final-holdout`, ir a la pestaña *Artifacts* > `model`, y usar la opción "Register Model" para crear una nueva versión en el Registry. Asignarle el alias *candidate*.
2. **Agregar tags a la versión registrada**: `decision_threshold = 0.45` y `final_holdout_run_id = d6de4f7b11424ee3990329dfd95a465e`, para dejar trazabilidad de qué umbral y qué run de evaluación final respaldan esa versión.
3. **Promover a *validation***: se reasigna el alias *validation* a esta versión una vez confirmado que su desempeño en `rf-final-holdout` (PR-AUC=0.46, Recall=0.72, F1=0.40) es consistente con lo observado previamente sobre `X_test_thr` (Experimento 15/17, mismo umbral).
4. **Promover a *production***: se reasigna el alias *production* tras validar que no hay regresión respecto a la comparación inicial entre los cuatro modelos (Experimentos 1-9) y que el equipo aprueba el resultado del holdout final.



---



## Docker

La imagen se construye asumiendo que el modelo ya fue exportado localmente (ver sección de *MLflow*).

### Dockerfile: decisiones documentadas

* **Base `python:3.10-slim`**: se evita la imagen "full" de Python para mantener el tamaño de la imagen razonable.
* **Orden de capas**: `requirements-api.txt` se copia e instala antes que el código.
* **`requirements-api.txt` en vez de `requirements.txt`**: la API no necesita `matplotlib`, `ucimlrepo`, `scipy`, u otros. Instalar solo lo que `app/main.py` importa para reducir el tamaño de la imagen.
* **Se copian tres carpetas**: `app/` (API), `src/` (porque `app/main.py` reutiliza `lower_case()` de `src/quality/clean.py`), y `model_artifact/` (modelo ya exportado + `metadata.json` con `model_version` y `decision_threshold`).
* **`HEALTHCHECK`**: permite que Docker (o un orquestador) determine si el servicio está realmente vivo, no solo si el proceso arrancó golpeando `/health`, que a su vez confirma que el modelo cargó correctamente.
* **`CMD` con `uvicorn` en `0.0.0.0`**: necesario para que el contenedor acepte conexiones externas al puerto mapeado (`--host 127.0.0.1` no funcionaría desde fuera del contenedor).


### Pre-requisito: exportar el modelo
Para continuar, se debe exportar el modelo ganador. Para ello ejecute el siguiente comando en la terminal: ​```python export_model.py​```


### Dependencias que utiliza docker
Archivo: ​```requirements-api.txt​```

```python
fastapi==0.115.0 
uvicorn[standard]==0.30.6
mlflow==3.15.1
scikit-learn==1.7.2
pyarrow==25.0.1 
pydantic==2.9.2
pandas==2.3.3
imbalanced-learn==0.14.2
```


### Build reproducible

1. Crear imagen <br>
​```docker build -t bank-marketing-api .​```

1. Crear contenedor de la imagen <br>
​```docker run -p 8000:8000 bank-marketing-api​```

1. Verificación
    1. En el navegador, ingresar en el buscador lo siguiente: http://localhost:8000/health
    2. Debe observar:
        ```json
       {"status": "ok", "model_version": "1"}
        ```
        Si da error http 503, debe revisar los logs del contenedor para ver el `load_error`.



---



## API

### Endpoints
* ​```GET /health​```: verifica que el servicio y el modelo cargaron correctamente
* ​```POST /predict​```: recibe features del cliente, devuelve predicción + probabilidad


### Levantar la API con Docker
1. Obtener el id del contenedor con el siguiente comando: ​```docker ps -a​```
2. Reemplaza '<container_id>' por el id del contenedor en el siguinte comando:  ```docker start <container_id>```
3. Ejecutar el comando en la terminal
4. Probar la API ingresando a http://localhost:8000/health


### Decisión de arquitectura: carga del modelo al iniciar, no por request
El modelo, su versión (`model_version`) y el umbral de decisión (`decision_threshold`) se leen una sola vez, al arrancar el proceso, no en cada llamada a `/predict`. Esto evita I/O repetido, haciendo la respuesta más rápida y predecible.

La carga está envuelta en un `try/except`:
- Si tiene éxito: `modelo`, `MODEL_VERSION` y `DECISION_THRESHOLD` quedan disponibles en memoria para toda la vida del proceso.
- Si falla: `modelo = None` y el motivo se guarda en `load_error`, en vez de que el contenedor crashee silenciosamente al arrancar. Esto permite que `/health` devuelva un `503` con el error real, en lugar de que la API quede en un estado ambiguo.


### Contrato de entrada (`ClientFeatures`)
El endpoint `/predict` espera un JSON con los siguientes 15 campos:

| Campo | Tipo | Restricción |
|---|---|---|
| age | int | ​```18 <= age <= 100​``` |
| job | str | — |
| marital | str | — |
| education | str | — |
| default | str | "yes" o "no" |
| balance | int | — |
| housing | str | "yes" o "no" |
| loan | str | "yes" o "no" |
| contact | str | — |
| day | int | ​```1 <= day <= 31​``` |
| month | str | — |
| campaign | int | >= 0 |
| pdays | int | >= -1 |
| previous | int | >= 0 |
| poutcome | str | — |

* **Por qué `Literal["yes", "no"]` en `default`, `housing`, `loan`**: cualquier valor fuera de ese dominio (ej. `"maybe"`, `"1"`, `""`) es rechazado automáticamente con un `422 Unprocessable Entity`, sin necesidad de validación manual en el endpoint. Esto es lo que cubren los casos de input inválido en `test_api.py`.

* **Nota sobre `alias` y `populate_by_name=True`**: en este modelo el `alias` de cada campo coincide con su nombre, por lo que actualmente no cambia el comportamiento. Se dejó explícito por si en el futuro el JSON de entrada usa nombres distintos a los atributos internos (ej. nombres con espacios o mayúsculas).


### Response (`PredictionResponse`)
```json
{
  "prediction": 0,
  "probability": 0.1234,
  "model_version": "1"
}
```

- `prediction`: clase predicha (`0` = no contrata, `1` = contrata), resultado de comparar `probability` contra `DECISION_THRESHOLD` (0.45).
- `probability`: probabilidad de la clase positiva (`"yes"`), redondeada a 4 decimales. Es la misma base sobre la que se fijó el umbral durante el análisis de threshold, no una probabilidad recalculada con otro criterio.
- `model_version`: versión del modelo en el MLflow Registry que generó la predicción, para trazabilidad.


### Ejemplo reproducible de request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35, "job": "technician", "marital": "married",
    "education": "professional.course", "default": "no", "balance": 1500,
    "housing": "yes", "loan": "no", "contact": "cellular",
    "day": 15, "month": "may", "campaign": 1,
    "pdays": -1, "previous": 0, "poutcome": "nonexistent"
  }'
```

### Punto abierto: `probability: float | None` ?? 
El tipo de `PredictionResponse.probability` permite `None`, pero el código actual siempre devuelve un `float` (nunca hay una rama que retorne `None`). Antes de dejarlo en el README como definitivo, conviene decidir:
- Si nunca puede ser `None` en la práctica → simplificar el tipo a `float` (más honesto con el contrato real).
- Si existe algún escenario donde sí debería ser `None` → documentarlo explícitamente aquí y agregar el caso a `test_api.py`.



---



## Monitoring

### System Monitoring
Correr docker
* python src/monitoring/monitor_api.py http://localhost:8000 20 15
* python src/monitoring/system_metrics.py

{
  "latency": {
    "mean_ms": 767.3,
    "p95_ms": 1096.15,
    "p99_ms": 9936.78
  },
  "throughput_requests_per_min": 2.58,
  "error_rate_pct": 2.5,
  "availability_pct": 97.5,
  "n_predict_calls": 40,
  "n_health_pings": 40
}

### Data Monitoring & Drift

Por desviacion estandar a los datos de balance y en job se fuerta "retired"

* python src/monitoring/data_drift.py

=== BATCH_1 (sin drift) ===
  age: PSI=0.0058 -> OK
  balance: PSI=0.0062 -> OK
  day: PSI=0.0084 -> OK
  campaign: PSI=0.0022 -> OK
  pdays: PSI=0.0005 -> OK
  previous: PSI=0.0009 -> OK
  job: PSI=0.007 -> OK
  marital: PSI=0.0008 -> OK
  education: PSI=0.0008 -> OK
  default: PSI=0.0001 -> OK
  housing: PSI=0.0001 -> OK
  loan: PSI=0.0007 -> OK
  contact: PSI=0.0002 -> OK
  month: PSI=0.0063 -> OK
  poutcome: PSI=0.001 -> OK

=== BATCH_2 (drift moderado) ===
  age: PSI=0.0068 -> OK
  balance: PSI=6.0278 -> ALERT
  day: PSI=0.0081 -> OK
  campaign: PSI=0.002 -> OK
  pdays: PSI=0.0002 -> OK
  previous: PSI=0.0006 -> OK
  job: PSI=0.4541 -> ALERT
  marital: PSI=0.0006 -> OK
  education: PSI=0.0014 -> OK
  default: PSI=0.0 -> OK
  housing: PSI=0.0001 -> OK
  loan: PSI=0.0003 -> OK
  contact: PSI=0.0007 -> OK
  month: PSI=0.0105 -> OK
  poutcome: PSI=0.0005 -> OK

=== BATCH_3 (drift fuerte) ===
  age: PSI=0.006 -> OK
  balance: PSI=8.0169 -> ALERT
  day: PSI=0.0049 -> OK
  campaign: PSI=0.0002 -> OK
  pdays: PSI=0.0002 -> OK
  previous: PSI=0.001 -> OK
  job: PSI=1.4291 -> ALERT
  marital: PSI=0.0009 -> OK
  education: PSI=0.0032 -> OK
  default: PSI=0.0002 -> OK
  housing: PSI=0.0 -> OK
  loan: PSI=0.0012 -> OK
  contact: PSI=0.0019 -> OK
  month: PSI=0.0062 -> OK
  poutcome: PSI=0.001 -> OK

### Model Monitoring

* python src/monitoring/model_monitor.py


---



## Results



---



## Team
El equipo se conforma por:

* **Sebastián Aguilar Benavides**: modelos KNN y Logistic Regression.
* **María Paula Elizondo Herrera**: modelos Decision Tree y Random Forest.

El resto de las secciones del proyecto (ingestión de datos, feature engineering, MLflow tracking/registry, análisis de umbral, evaluación del modelo ganador, Docker, API y testing) fueron desarrolladas en conjunto por ambos integrantes.



---



## Comandos reproducibles

### Entorno virtual
* python -m venv mlflow-project
* mlflow-project/Scripts/activate
* pip install -r requirements.txt


### Datos
* python src/ingestion/ingest.py
* python src/quality/clean.py
* python src/quality/gates.py
* python tests/test_clean_gates.py


### MLflow
* mlflow server --backend-store-uri sqlite:///bank.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000


### Experimentos y modelos
* python src/training.py
* python src/evalution/threshold_analysis.py
* python export_model.py


### Dockers
* docker build -t bank-marketing-api .
* docker run -p 8000:8000 bank-marketing-api


### Testing
* pytest tests/test_api.py -v
* pytest tests/test_data.py -v
* pytest tests/test_model.py -v

O bien, para correr todos:

* pytest tests/ -v

---
