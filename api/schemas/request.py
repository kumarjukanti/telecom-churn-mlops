
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChurnRequest(BaseModel):
    """Business-level customer features for churn prediction."""

    model_config = ConfigDict(extra="forbid")

    age: int | None = Field(default=None, ge=18, le=110)

    gender: Literal["Female", "Male"]
    region: Literal["Rural", "Suburban", "Urban"]
    contract_type: Literal["Monthly", "Prepaid", "Yearly"]

    monthly_charges: float = Field(ge=0)
    months_observed: int = Field(ge=1)

    avg_call_minutes: float = Field(ge=0)
    avg_data_usage_gb: float = Field(ge=0)
    avg_sms_count: float = Field(ge=0)

    total_complaints: float = Field(ge=0)
    avg_complaints: float = Field(ge=0)

    tenure_months: float = Field(ge=0)
    complaint_rate: float = Field(ge=0)
    charges_per_tenure_month: float = Field(ge=0)
