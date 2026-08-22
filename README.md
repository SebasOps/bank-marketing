# bank-marketing

## ...

## Ingesta de datos

El dataset (Bank Marketing, UCI ML Repository, id=222) se obtiene mediante:

​```bash
python src/ingestion/ingest.py
​```

Esto descarga los datos directamente desde UCI usando la librería `ucimlrepo` y los guarda en `data/raw/bank_marketing.csv`. No se requiere ningún archivo local previo. Script es completamente reproducible.

### Requisitos
- Instalar dependencias: `pip install -r requirements.txt`