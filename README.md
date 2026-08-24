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

---

## Installation

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
* María Paula ...

Donde cada uno de los integrantes aportó todas y cada una de las diferentes secciones.

---

## Comandos reproducibles 

* git clone https://github.com/SebasOps/bank-marketing/
* pip install -r requirements.tx
* python src/ingestion/ingest.py
* 
* 

---
---
# Notas:
DESCARTAR 'duration' DEL MODELO
