"""
Monitoreo black-box de la API: mide Latency, Throughput, Error Rate (via /predict)
y Availability (via /health) llamando a la API real desde afuera.
Genera un único log: logs/api_monitor.jsonl
"""
import sys, time, json, requests
from pathlib import Path
from datetime import datetime, timezone

LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "api_monitor.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)

# Payload de ejemplo representativo de un cliente real
SAMPLE_PAYLOAD = {
    "age": 35, "job": "technician", "marital": "married", "education": "secondary",
    "default": "no", "balance": 1500, "housing": "yes", "loan": "no",
    "contact": "cellular", "day": 15, "month": "may", "campaign": 2,
    "pdays": -1, "previous": 0, "poutcome": "unknown"
}

def log(entry: dict):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def call_health(api_url):
    start = time.perf_counter()
    try:
        r = requests.get(f"{api_url}/health", timeout=10)
        ok = r.status_code == 200
        status_code = r.status_code
    except requests.RequestException:
        ok = False
        status_code = 0
    log({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": "/health",
        "status_code": status_code,
        "available": ok,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    })

def call_predict(api_url):
    start = time.perf_counter()
    try:
        r = requests.post(f"{api_url}/predict", json=SAMPLE_PAYLOAD, timeout=15)
        status_code = r.status_code
    except requests.RequestException:
        status_code = 0
    log({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": "/predict",
        "status_code": status_code,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    })

if __name__ == "__main__":
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    n_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    interval = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    for i in range(n_cycles):
        call_health(api_url)
        call_predict(api_url)
        print(f"ciclo {i+1}/{n_cycles} loggeado")
        time.sleep(interval)