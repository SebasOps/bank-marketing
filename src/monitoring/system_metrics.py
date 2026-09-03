import json
import numpy as np
from pathlib import Path
from datetime import datetime


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_system_metrics(log_dir: Path):
    """
    Calcula Latency, Throughput y Error Rate a partir de requests.jsonl.
    Corre DENTRO del contenedor (expuesto vía /metrics), porque requests.jsonl
    solo existe ahí.
    """
    requests_log = load_jsonl(log_dir / "requests.jsonl")
    predict_requests = [r for r in requests_log if r["endpoint"] == "/predict"]

    if not predict_requests:
        return {
            "latency": None,
            "throughput_requests_per_min": None,
            "error_rate_pct": None,
            "n_requests_analyzed": 0,
        }

    latencies = [r["latency_ms"] for r in predict_requests]
    errors = [r for r in predict_requests if r["status_code"] >= 400]
    timestamps = [datetime.fromisoformat(r["timestamp"]) for r in predict_requests]
    span_min = max((max(timestamps) - min(timestamps)).total_seconds() / 60, 1 / 60)

    return {
        "latency": {
            "mean_ms": round(float(np.mean(latencies)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        },
        "throughput_requests_per_min": round(len(predict_requests) / span_min, 2),
        "error_rate_pct": round(len(errors) / len(predict_requests) * 100, 2),
        "n_requests_analyzed": len(predict_requests),
    }


def compute_availability(health_log_path: Path):
    """
    Calcula Availability a partir de health_pings.jsonl.
    Corre LOCAL (nunca dentro del contenedor: el ping tiene que venir de afuera
    para que mida disponibilidad real, no que el servicio se audite a sí mismo).
    """
    health_log = load_jsonl(health_log_path)
    if not health_log:
        return {"availability_pct": None, "n_pings": 0}

    return {
        "availability_pct": round(sum(h["available"] for h in health_log) / len(health_log) * 100, 2),
        "n_pings": len(health_log),
    }

if __name__ == "__main__":
    # Uso local: python src/monitoring/system_metrics.py
    LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
    print("System metrics (requiere requests.jsonl LOCAL, normalmente solo vía /metrics en Render):")
    print(json.dumps(compute_system_metrics(LOG_DIR), indent=2))
    print("\nAvailability (local):")
    print(json.dumps(compute_availability(LOG_DIR / "health_pings.jsonl"), indent=2))