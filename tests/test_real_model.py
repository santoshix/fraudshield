"""The real-data benchmark is optional and should not break the default demo."""
import numpy as np
import pytest
from ml.train_real import FEATURES
from app.real_risk import score_real_transaction


def test_real_features_schema():
    assert len(FEATURES) == 30
    assert FEATURES[0] == "Time" and FEATURES[-1] == "Amount"


def test_missing_features_rejected():
    with pytest.raises(ValueError, match="Exactly these features"):
        score_real_transaction({"Amount": 100})


def test_nonfinite_inputs_rejected_without_loading_model():
    features = dict.fromkeys(FEATURES, 0)
    features["Amount"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        score_real_transaction(features)



def test_published_historical_dashboard_report():
    from fastapi.testclient import TestClient
    from app.main import app
    # Avoid triggering application startup: this report endpoint needs no database.
    client = TestClient(app)
    response = client.get("/api/historical-benchmark")
    assert response.status_code == 200
    metrics = response.json()
    assert metrics["rows"] == 283726
    assert metrics["test_rows"] == 56746
    assert metrics["roc_auc"] == 0.9815


def test_historical_samples_requires_local_files(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app.main import app
    import ml.train_real as training
    monkeypatch.setattr(training, "DATA_PATH", tmp_path / "missing.csv")
    # This endpoint returns before accessing the database; no startup needed.
    client = TestClient(app)
    response = client.get("/api/historical-samples")
    assert response.status_code == 503
    assert "not installed" in response.json()["detail"]
