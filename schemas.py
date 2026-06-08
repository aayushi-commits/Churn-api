from pydantic import BaseModel, Field
from typing import Optional


class CustomerInput(BaseModel):
    gender: str = Field(..., examples=["Male"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["DSL"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[29.85])
    TotalCharges: float = Field(..., ge=0, examples=[29.85])


class TopReason(BaseModel):
    feature: str
    contribution: float
    direction: str  # "increases" or "decreases"


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_percent: str
    risk_tier: str          # low / medium / high
    top_reasons: list[TopReason]
    model_version: str
    model_name: str


class RetrainResponse(BaseModel):
    status: str
    new_auc: float
    champion_auc: Optional[float]
    promoted: bool
    message: str


class ModelInfo(BaseModel):
    model_name: str
    version: str
    stage: str
    auc: Optional[float]
    f1: Optional[float]
    training_date: Optional[str]
    model_type: Optional[str]
    run_id: Optional[str]


class RunInfo(BaseModel):
    run_id: str
    model_type: Optional[str]
    auc: Optional[float]
    f1: Optional[float]
    timestamp: Optional[str]
    status: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str]
    model_version: Optional[str]
    uptime_seconds: float
