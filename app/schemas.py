from pydantic import BaseModel, Field, ConfigDict

class TransactionInput(BaseModel):
    amount: float = Field(ge=0.01, le=1_000_000)
    hour: int = Field(ge=0, le=23)
    distance_from_home: float = Field(ge=0, le=30_000)
    is_foreign: bool = False
    is_online: bool = True
    velocity_24h: int = Field(ge=0, le=100)
    account_age_days: int = Field(ge=0, le=36500)
    merchant_risk: float = Field(ge=0, le=1)

    model_config = ConfigDict(json_schema_extra={
        "example": {"amount": 2450, "hour": 2, "distance_from_home": 650,
                    "is_foreign": True, "is_online": True, "velocity_24h": 11,
                    "account_age_days": 14, "merchant_risk": 0.82}
    })
