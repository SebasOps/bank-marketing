import json
import numpy as np
from pathlib import Path
from datetime import datetime


LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


def load_jsonl(path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_system_metrics():
    requests_log = load_jsonl(LOG_DIR / "requests.jsonl")
    health_log = load_jsonl(LOG_DIR / "health_pings.jsonl")
    predict_requests = [r for r in requests_log if r["endpoint"] == "/predict"]

    if predict_requests:
        latencies = [r["latency_ms"] for r in predict_requests]
        errors = [r for r in predict_requests if r["status_code"] >= 400]
        timestamps = [datetime.fromisoformat(r["timestamp"]) for r in predict_requests]
        span_min = max((max(timestamps) - min(timestamps)).total_seconds() / 60, 1/60)
        latency_metrics = {
            "mean_ms": round(float(np.mean(latencies)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        }
        throughput_rpm = round(len(predict_requests) / span_min, 2)
        error_rate = round(len(errors) / len(predict_requests) * 100, 2)
    else:
        latency_metrics, throughput_rpm, error_rate = None, None, None

    if health_log:
        availability_pct = round(sum(h["available"] for h in health_log) / len(health_log) * 100, 2)
    else:
        availability_pct = None

    return {
        "latency": latency_metrics,
        "throughput_requests_per_min": throughput_rpm,
        "error_rate_pct": error_rate,
        "availability_pct": availability_pct,
        "n_requests_analyzed": len(predict_requests),
        "n_health_pings": len(health_log),
    }


if __name__ == "__main__":
    print(json.dumps(compute_system_metrics(), indent=2))