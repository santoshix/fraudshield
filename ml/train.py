"""Train a real XGBoost classifier on synthetic (made-up!) transactions.

DO NOT use this model to make real financial decisions. Fraud labels are simulated.
Replace this generator with properly licensed/authorized real data for a research study.
"""
import json
import os
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

FEATURES = [
    "amount", "hour", "distance_from_home", "is_foreign", "is_online",
    "velocity_24h", "account_age_days", "merchant_risk"
]
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_FILE = MODEL_DIR / "fraud_xgboost.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"

def make_synthetic_data(n=11000, seed=42):
    rng = np.random.default_rng(seed)
    amount = np.clip(rng.lognormal(mean=4.9, sigma=1.25, size=n), 2, 10000)
    hour = rng.integers(0, 24, size=n)
    distance = np.clip(rng.exponential(scale=67, size=n), 0, 2500)
    foreign = rng.binomial(1, 0.095, size=n)
    online = rng.binomial(1, 0.58, size=n)
    velocity = np.clip(rng.poisson(2.4, size=n) + rng.binomial(1, .035, n) * rng.integers(4, 14, n), 0, 100)
    age = np.clip(rng.exponential(scale=560, size=n), 0, 36500).astype(int)
    merchant_risk = rng.beta(2, 5, size=n)
    # This scoring logic is invented for a teaching exercise, NOT a banking rule.
    logits = (-5.1 + .00042 * amount + .0032 * distance + 1.3 * foreign
              + .26 * velocity + .9 * merchant_risk + .65 * (age < 30)
              + .65 * (hour < 5) + .35 * online)
    probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    label = rng.binomial(1, probs)
    X = np.column_stack([amount, hour, distance, foreign, online, velocity, age, merchant_risk])
    return X, label

def train_model(n=11000):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y = make_synthetic_data(n=n)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.24, random_state=42, stratify=y)
    model = XGBClassifier(
        n_estimators=170, max_depth=4, learning_rate=.065,
        subsample=.9, colsample_bytree=.9, objective="binary:logistic",
        eval_metric="logloss", random_state=42, n_jobs=2,
    )
    model.fit(Xtrain, ytrain)
    p = model.predict_proba(Xtest)[:, 1]
    pred = (p >= .35).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(ytest, pred, average="binary", zero_division=0)
    metrics = {
        "model": "XGBoost", "version": "demo-1.0", "dataset": "Synthetic (not real banking data)",
        "samples": int(n), "training_examples": int(len(Xtrain)),
        "validation_examples": int(len(Xtest)), "simulated_fraud_rate": round(float(y.mean() * 100), 2),
        "roc_auc": round(float(roc_auc_score(ytest, p)), 3),
        "average_precision": round(float(average_precision_score(ytest, p)), 3),
        "precision_at_review_threshold": round(float(precision), 3),
        "recall_at_review_threshold": round(float(recall), 3),
        "f1_at_review_threshold": round(float(f1), 3),
        "threshold_review": 35, "threshold_high": 70,
        "features": FEATURES,
    }
    joblib.dump(model, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    print(f"Trained XGBoost on {n:,} SYNTHETIC transactions | ROC AUC: {metrics['roc_auc']}")
    return metrics

if __name__ == "__main__":
    train_model()
