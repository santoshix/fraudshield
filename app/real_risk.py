"""Separate inference for the optional ULB historical-fraud benchmark."""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from ml.train_real import FEATURES, METRICS_PATH, MODEL_PATH


@lru_cache(maxsize=1)
def load_real_model():
    if not MODEL_PATH.is_file() or not METRICS_PATH.is_file():
        raise FileNotFoundError("Real-data model not trained; run python -m ml.train_real")
    model = joblib.load(MODEL_PATH)  # Load only local artifacts you trust.
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return model, metrics


def score_real_transaction(features):
    if set(features) != set(FEATURES):
        raise ValueError(f"Exactly these features are required: {', '.join(FEATURES)}")
    try:
        values = np.array([[float(features[k]) for k in FEATURES]], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Features must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite")
    model, metrics = load_real_model()
    probability = float(model.predict_proba(values)[0, 1])
    threshold = float(metrics["decision_threshold"])
    return {"fraud_probability": round(probability, 6),
            "flagged_for_review": probability >= threshold,
            "decision_threshold": threshold,
            "note": "Historical research benchmark, not a financial decision."}
