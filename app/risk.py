"""Load trained model and score transaction feature vectors."""
import json
from functools import lru_cache
import joblib
import numpy as np
from ml.train import MODEL_FILE, METRICS_FILE, train_model
from app.schemas import TransactionInput

@lru_cache(maxsize=1)
def get_model():
    if not MODEL_FILE.exists() or not METRICS_FILE.exists():
        train_model()
    return joblib.load(MODEL_FILE)

def get_model_metrics():
    get_model()
    return json.loads(METRICS_FILE.read_text())

def predict_risk(txn: TransactionInput):
    model = get_model()
    features = np.array([[
        txn.amount, txn.hour, txn.distance_from_home, int(txn.is_foreign),
        int(txn.is_online), txn.velocity_24h, txn.account_age_days,
        txn.merchant_risk,
    ]], dtype=float)
    score = round(float(model.predict_proba(features)[0, 1] * 100), 2)
    status = "HIGH RISK" if score >= 70 else "REVIEW" if score >= 35 else "LOW RISK"
    return score, status
