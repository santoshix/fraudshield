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
