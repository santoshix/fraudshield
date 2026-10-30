"""FastAPI service: fraud scoring, monitoring, demo records and Kafka ingestion."""
import json
import os
import random
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, func, case
from sqlalchemy.orm import Session

from app.db import Transaction, SessionLocal, get_db, init_db
from app.risk import get_model, get_model_metrics, predict_risk
from app.schemas import TransactionInput

STATIC = Path(__file__).parent / "static"

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_model()   # auto-train locally the first time you start
    yield

app = FastAPI(title="FraudShield | Fraud Detection & Risk Engine", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC / "index.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "model": "XGBoost", "database": "connected", "kafka_enabled": bool(os.getenv("KAFKA_BOOTSTRAP_SERVERS"))}

@app.get("/api/model")
def model_info():
    return get_model_metrics()

def serialize(t: Transaction):
    return {
        "transaction_id": t.transaction_id, "amount": t.amount, "hour": t.hour,
        "distance_from_home": t.distance_from_home, "is_foreign": t.is_foreign,
        "is_online": t.is_online, "velocity_24h": t.velocity_24h,
        "account_age_days": t.account_age_days, "merchant_risk": t.merchant_risk,
        "risk_score": t.risk_score, "status": t.status, "source": t.source,
        "created_at": t.created_at.replace(tzinfo=timezone.utc).isoformat() if t.created_at.tzinfo is None else t.created_at.isoformat(),
    }

def save_transaction(data: TransactionInput, db: Session, source="API", created_at=None):
    score, status = predict_risk(data)
    record = Transaction(
        transaction_id="TXN-" + uuid.uuid4().hex[:10].upper(),
        **data.model_dump(), risk_score=score, status=status, source=source,
        created_at=created_at or datetime.now(timezone.utc),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return serialize(record)

@app.post("/api/score")
def score_transaction(data: TransactionInput, db: Session = Depends(get_db)):
    return save_transaction(data, db)

@app.get("/api/transactions")
def transactions(limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    results = db.scalars(select(Transaction).order_by(Transaction.created_at.desc(), Transaction.id.desc()).limit(limit)).all()
    return [serialize(t) for t in results]

@app.get("/api/metrics")
def metrics(db: Session = Depends(get_db)):
    count, high, review, avg, volume = db.execute(
        select(func.count(Transaction.id),
               func.coalesce(func.sum(case((Transaction.status == "HIGH RISK", 1), else_=0)), 0),
               func.coalesce(func.sum(case((Transaction.status == "REVIEW", 1), else_=0)), 0),
               func.coalesce(func.avg(Transaction.risk_score), 0.0),
               func.coalesce(func.sum(Transaction.amount), 0.0))
    ).one()
    total = int(count)
    recent = db.scalars(select(Transaction).order_by(Transaction.created_at.desc(), Transaction.id.desc()).limit(100)).all()
    return {
        "total": total, "high_risk": int(high), "review": int(review),
        "low_risk": total - int(high) - int(review),
        "flagged_percent": round(((int(high) + int(review)) / total) * 100, 1) if total else 0,
        "average_risk": round(float(avg or 0), 1),
        "total_volume": round(float(volume or 0), 2),
        "recent_scores": [{"score": t.risk_score, "time": serialize(t)["created_at"]} for t in reversed(recent)]
    }

@app.post("/api/demo")
def create_demo_transactions(count: int = Query(55, ge=1, le=300), db: Session = Depends(get_db)):
    rng = random.Random(2026 + count)
    statuses = {"HIGH RISK": 0, "REVIEW": 0, "LOW RISK": 0}
    for i in range(count):
        suspicious = rng.random() < 0.27
        if suspicious:
            sample = TransactionInput(
                amount=round(rng.uniform(900, 7600), 2), hour=rng.choice([0, 1, 2, 3, 4, 23]),
                distance_from_home=round(rng.uniform(280, 1600), 1), is_foreign=rng.random() < .75,
                is_online=True, velocity_24h=rng.randint(6, 21), account_age_days=rng.randint(1, 90),
                merchant_risk=round(rng.uniform(.55, .98), 2),
            )
        else:
            sample = TransactionInput(
                amount=round(rng.uniform(8, 650), 2), hour=rng.randint(7, 22),
                distance_from_home=round(rng.uniform(1, 110), 1), is_foreign=rng.random() < .07,
                is_online=rng.random() < .62, velocity_24h=rng.randint(0, 5),
                account_age_days=rng.randint(65, 2300), merchant_risk=round(rng.uniform(.03, .5), 2),
            )
        created = datetime.now(timezone.utc) - timedelta(minutes=rng.randint(0, 20 * 60))
        result = save_transaction(sample, db, "DEMO", created_at=created)
        statuses[result["status"]] += 1
    return {"created": count, "status_breakdown": statuses,
            "note": "All generated records are artificial, and fraud predictions are not verified labels."}

@app.post("/api/ingest")
def ingest_transaction(data: TransactionInput, db: Session = Depends(get_db)):
    """Submit to Kafka if configured, otherwise score directly (local quick-start mode)."""
    brokers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "").strip()
    if not brokers:
        return {"mode": "direct", "result": save_transaction(data, db, "DIRECT")}
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=brokers.split(","),
            value_serializer=lambda x: json.dumps(x).encode("utf-8"),
            request_timeout_ms=6000, max_block_ms=6000,
        )
        producer.send("fraud.incoming", data.model_dump()).get(timeout=8)
        producer.flush(timeout=8)
        producer.close(timeout=5)
        return {"mode": "kafka", "status": "queued", "note": "Kafka worker will score and store the transaction shortly."}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Kafka is not reachable. Check the Docker Kafka service.") from exc
