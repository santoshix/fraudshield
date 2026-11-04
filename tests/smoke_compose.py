"""End-to-end smoke check for the Docker Kafka -> worker -> PostgreSQL path.

Run only against a local Docker Compose instance, never a production service.
"""
import json
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8000"
SAMPLE = {
    "amount": 431.27, "hour": 2, "distance_from_home": 600,
    "is_foreign": True, "is_online": True, "velocity_24h": 9,
    "account_age_days": 7, "merchant_risk": 0.85,
}


def request(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=20) as response:
        return json.load(response)


def await_condition(fn, seconds, description):
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        try:
            value = fn()
            if value:
                return value
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            last = str(exc)
        time.sleep(3)
    raise AssertionError(f"Timed out waiting for {description}; last error: {last}")


def main():
    health = await_condition(
        lambda: request("/api/health") if request("/api/health").get("status") == "ok" else None,
        240, "API health",
    )
    assert health["kafka_enabled"], "Docker mode must have Kafka configured"
    request("/api/model")  # ensure inference/model startup is available
    before = request("/api/metrics")["total"]

    queued = await_condition(
        lambda: request("/api/ingest", SAMPLE),
        90, "Kafka producer accepting a transaction",
    )
    assert queued["mode"] == "kafka" and queued["status"] == "queued", queued

    def persisted():
        rows = request("/api/transactions?limit=500")
        found = [row for row in rows if row["source"] == "KAFKA" and row["amount"] == SAMPLE["amount"]]
        return found[0] if found else None

    row = await_condition(persisted, 180, "Kafka worker persisting the scored event")
    assert row["status"] in ("HIGH RISK", "REVIEW", "LOW RISK")
    assert 0 <= row["risk_score"] <= 100
    after = request("/api/metrics")["total"]
    assert after >= before + 1, (before, after)
    print(json.dumps({
        "result": "PASS", "pipeline": "FastAPI -> Kafka -> worker -> PostgreSQL",
        "transaction_id": row["transaction_id"], "score": row["risk_score"],
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SMOKE TEST FAILED: {exc}", file=sys.stderr)
        raise
