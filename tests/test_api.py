"""Run with python -m pytest -q (uses an isolated temporary SQLite database)."""
import os
import tempfile
from pathlib import Path

_temp_dir = tempfile.TemporaryDirectory()
os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(_temp_dir.name) / "test.db")
os.environ.pop("KAFKA_BOOTSTRAP_SERVERS", None)

from fastapi.testclient import TestClient
from app.main import app

SAMPLE = { 
    "amount": 2500.0, "hour": 2, "distance_from_home": 740.0,
    "is_foreign": True, "is_online": True, "velocity_24h": 12,
    "account_age_days": 3, "merchant_risk": .9,
}

def test_healthy_and_predict():
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        r = client.post("/api/score", json=SAMPLE)
        assert r.status_code == 200, r.text
        txn = r.json()
        assert 0 <= txn["risk_score"] <= 100
        assert txn["status"] in {"HIGH RISK", "REVIEW", "LOW RISK"}
        assert txn["transaction_id"].startswith("TXN-")
        assert client.get("/api/metrics").json()["total"] >= 1
        assert len(client.get("/api/transactions").json()) >= 1

def test_invalid_amount_rejected():
    with TestClient(app) as client:
        r = client.post("/api/score", json={**SAMPLE, "amount": -4})
        assert r.status_code == 422

def test_demo_generator():
    with TestClient(app) as client:
        r = client.post("/api/demo?count=4")
        assert r.status_code == 200
        assert r.json()["created"] == 4
def teardown_module():
    from app.db import engine

    engine.dispose()
    _temp_dir.cleanup() 

def test_kafka_event_deduplication():
    from app.main import save_transaction
    from app.db import SessionLocal
    from app.schemas import TransactionInput
    with TestClient(app):
        with SessionLocal() as db:
            data = TransactionInput.model_validate(SAMPLE)
            transaction_id = "EVT-" + "a" * 32
            first = save_transaction(data, db, "KAFKA", transaction_id=transaction_id)
            second = save_transaction(data, db, "KAFKA", transaction_id=transaction_id)
            assert first["transaction_id"] == second["transaction_id"]
            assert first["risk_score"] == second["risk_score"]


def test_api_key_optional_protection(monkeypatch):
    monkeypatch.setenv("FRAUDSHIELD_API_KEY", "test-secret")
    with TestClient(app) as client:
        assert client.post("/api/score", json=SAMPLE).status_code == 401
        assert client.post("/api/score", json=SAMPLE, headers={"X-API-Key": "test-secret"}).status_code == 200
