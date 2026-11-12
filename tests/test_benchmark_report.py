"""Regression checks for the published historical-fraud evaluation."""
import json
from pathlib import Path

from ml.train_real import FEATURES

REPORT = Path(__file__).resolve().parents[1] / "benchmarks" / "real_metrics.json"


def test_published_benchmark_structure():
    results = json.loads(REPORT.read_text(encoding="utf-8"))
    assert results["rows"] == (
        results["train_rows"] + results["validation_rows"] + results["test_rows"]
    )
    assert results["features"] == FEATURES
    assert 0 <= results["precision"] <= 1
    assert 0 <= results["recall"] <= 1
    assert 0 <= results["average_precision"] <= 1
    assert 0 <= results["roc_auc"] <= 1
    assert len(results["walk_forward_cv_roc_auc"]) == 3
    assert len(results["walk_forward_cv_average_precision"]) == 3
    assert 0 < results["decision_threshold"] < 1
