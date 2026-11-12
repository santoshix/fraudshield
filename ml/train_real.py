"""Optional real-world fraud benchmark using the public ULB credit-card dataset.

Requires data/creditcard.csv downloaded separately from Kaggle. This does NOT
replace the eight-feature synthetic demo model or its existing dashboard.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "creditcard.csv"
MODEL_PATH = ROOT / "models" / "fraud_real_xgboost.joblib"
METRICS_PATH = ROOT / "models" / "real_metrics.json"
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def train_real_dataset(csv_path=DATA_PATH):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing {csv_path}. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in the data/ folder."
        )
    df = pd.read_csv(csv_path)
    required = set(FEATURES + ["Class"])
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    df = df[FEATURES + ["Class"]].dropna().drop_duplicates()
    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite feature values detected")
    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class must only contain 0 and 1")
    # Chronological holdout reduces future-to-past leakage.
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(df)
    if n < 100:
        raise ValueError("Dataset too small for this benchmark")
    first, second = int(n * 0.6), int(n * 0.8)
    train, val, test = df.iloc[:first], df.iloc[first:second], df.iloc[second:]
    if any(part["Class"].nunique() < 2 for part in (train, val, test)):
        raise ValueError("Every chronological split needs fraud and non-fraud examples")
    X_train, y_train = train[FEATURES], train["Class"]
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=float((y_train == 0).sum() / (y_train == 1).sum()),
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=2,
    )
    model.fit(X_train, y_train)
    val_probs = model.predict_proba(val[FEATURES])[:, 1]
    thresholds = np.arange(0.05, 0.96, 0.05)
    threshold = float(max(
        thresholds,
        key=lambda t: f1_score(val["Class"], val_probs >= t, zero_division=0),
    ))
    test_probs = model.predict_proba(test[FEATURES])[:, 1]
    predicted = test_probs >= threshold
    metrics = {
        "dataset": "ULB creditcard.csv (historical anonymized transactions)",
        "chronological_split": "60% train / 20% validation / 20% test",
        "rows": n,
        "train_rows": len(train), "validation_rows": len(val),
        "test_rows": len(test),
        "test_fraud_rate": round(float(test["Class"].mean()), 6),
        "decision_threshold": round(threshold, 3),
        "roc_auc": round(float(roc_auc_score(test["Class"], test_probs)), 4),
        "average_precision": round(float(average_precision_score(test["Class"], test_probs)), 4),
        "precision": round(float(precision_score(test["Class"], predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(test["Class"], predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(test["Class"], predicted, zero_division=0)), 4),
        "features": FEATURES,
        "limitations": "Historical anonymized labels; not validated on current payments.",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_real_dataset(), indent=2))
