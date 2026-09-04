# Imports 
import json
import numpy as np
from pathlib import Path
from datetime import datetime


def load_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_system_metrics(log_path: Path):
    entries = load_jsonl(log_path)
    predict_entries = [e for e in entries if e["endpoint"] == "/predict"]
    health_entries = [e for e in entries if e["endpoint"] == "/health"]

    result = {"latency": None, "throughput_requests_per_min": None,
               "error_rate_pct": None, "availability_pct": None,
               "n_predict_calls": len(predict_entries), "n_health_pings": len(health_entries)}

    if predict_entries:
        latencies = [e["latency_ms"] for e in predict_entries]
        errors = [e for e in predict_entries if e["status_code"] >= 400 or e["status_code"] == 0]
        timestamps = [datetime.fromisoformat(e["timestamp"]) for e in predict_entries]
        span_min = max((max(timestamps) - min(timestamps)).total_seconds() / 60, 1 / 60)

        result["latency"] = {
            "mean_ms": round(float(np.mean(latencies)), 2),
            "p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        }
        result["throughput_requests_per_min"] = round(len(predict_entries) / span_min, 2)
        result["error_rate_pct"] = round(len(errors) / len(predict_entries) * 100, 2)

    if health_entries:
        result["availability_pct"] = round(
            sum(e["available"] for e in health_entries) / len(health_entries) * 100, 2
        )

    return result


if __name__ == "__main__":
    LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "api_monitor.jsonl"
    print(json.dumps(compute_system_metrics(LOG_PATH), indent=2))