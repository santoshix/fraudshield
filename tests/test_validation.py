"""Fast, model-independent checks of public request validation."""
from pydantic import ValidationError
import pytest
from app.schemas import TransactionInput

VALID = dict(
    amount=100.0, hour=12, distance_from_home=2.0,
    is_foreign=False, is_online=True, velocity_24h=1,
    account_age_days=365, merchant_risk=0.2,
)

@pytest.mark.parametrize("field,value", [
    ("amount", 0), ("amount", 1_000_001),
    ("hour", -1), ("hour", 24),
    ("distance_from_home", -1), ("velocity_24h", 101),
    ("account_age_days", -1), ("merchant_risk", -0.1),
    ("merchant_risk", 1.1),
])
def test_rejects_out_of_range_input(field, value):
    with pytest.raises(ValidationError):
        TransactionInput(**(VALID | {field: value}))

def test_accepts_valid_input():
    assert TransactionInput(**VALID).amount == 100.0
