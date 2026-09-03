import sys, time, json, requests
from pathlib import Path
from datetime import datetime, timezone


API_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "health_pings.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True)


def ping():
    start = time.perf_counter()
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        ok = r.status_code == 200
    except requests.RequestException:
        ok = False
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "available": ok,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    n_pings = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    interval = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    for i in range(n_pings):
        ping()
        print(f"ping {i+1}/{n_pings} logged")
        time.sleep(interval)
        