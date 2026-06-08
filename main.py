"""
main.py — FastAPI churn prediction API.
Endpoints: POST /predict, POST /retrain, GET /model/info, GET /model/runs, GET /health
"""

import os
import time
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, HTMLResponse
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from schemas import (
    CustomerInput,
    PredictionResponse,
    TopReason,
    RetrainResponse,
    ModelInfo,
    RunInfo,
    HealthResponse,
)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "ChurnModel")
MODEL_STAGE = "Production"

# ── App state ─────────────────────────────────────────────────────────────────
_state: dict = {
    "pipeline": None,
    "model_version": None,
    "start_time": time.time(),
}


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_production_model() -> tuple:
    """Load the Production model from MLflow. Returns (pipeline, version_str)."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        return None, None
    v = versions[0]
    pipeline = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    return pipeline, str(v.version)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        pipeline, version = _load_production_model()
        _state["pipeline"] = pipeline
        _state["model_version"] = version
        if pipeline:
            print(f"Loaded {MODEL_NAME} v{version} from Production.")
        else:
            print(f"No Production model found for '{MODEL_NAME}'. Run train.py first.")
    except Exception as e:
        print(f"Model load warning: {e}")
    yield
    # cleanup (nothing needed)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Predicts customer churn probability using a Logistic Regression pipeline. "
        "Returns probability, risk tier, and the top 3 coefficient-based reasons."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def _customer_to_df(customer: CustomerInput) -> pd.DataFrame:
    data = customer.model_dump()
    return pd.DataFrame([data])[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]


def _risk_tier(prob: float) -> str:
    if prob < 0.35:
        return "low"
    elif prob < 0.65:
        return "medium"
    return "high"


def _get_top_reasons(pipeline, input_df: pd.DataFrame, top_n: int = 3) -> list[TopReason]:
    """
    Coefficient-based explainability:
      contribution_i = coef_i × scaled_feature_value_i
    Top 3 by absolute contribution.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["classifier"]

    X_transformed = preprocessor.transform(input_df)  # shape (1, n_features)
    coef = clf.coef_[0]                                # shape (n_features,)
    contributions = coef * X_transformed[0]

    feature_names = preprocessor.get_feature_names_out()

    # Clean up transformer prefixes (num__, cat__)
    clean_names = [
        n.replace("num__", "").replace("cat__", "") for n in feature_names
    ]

    indices = np.argsort(np.abs(contributions))[::-1][:top_n]

    reasons = []
    for idx in indices:
        contrib = float(contributions[idx])
        reasons.append(
            TopReason(
                feature=clean_names[idx],
                contribution=round(contrib, 4),
                direction="increases churn risk" if contrib > 0 else "decreases churn risk",
            )
        )
    return reasons


def _get_run_metric(client: MlflowClient, run_id: str, key: str) -> Optional[float]:
    try:
        return client.get_run(run_id).data.metrics.get(key)
    except Exception:
        return None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return HTMLResponse(content=(BASE_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerInput):
    """Score a single customer — returns churn probability, risk tier, and top 3 reasons."""
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="No Production model loaded. Run train.py to train and register one.",
        )

    input_df = _customer_to_df(customer)
    prob = float(pipeline.predict_proba(input_df)[0, 1])
    tier = _risk_tier(prob)
    reasons = _get_top_reasons(pipeline, input_df)

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_percent=f"{prob * 100:.1f}%",
        risk_tier=tier,
        top_reasons=reasons,
        model_version=_state.get("model_version") or "unknown",
        model_name=MODEL_NAME,
    )


@app.post("/retrain", response_model=RetrainResponse, tags=["Training"])
def retrain(background_tasks: BackgroundTasks):
    """
    Trigger a full retrain on the Telco dataset.
    Compares new AUC vs current Production champion.
    Promotes to Production only if improvement > 0.005.
    """
    # Import here to avoid circular deps and keep startup fast
    from train import train as run_train

    try:
        result = run_train()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")

    # Reload the production model if it was promoted
    if result["promoted"]:
        try:
            pipeline, version = _load_production_model()
            _state["pipeline"] = pipeline
            _state["model_version"] = version
        except Exception:
            pass  # non-fatal — next request will still use old model

    champion = result.get("champion_auc")
    new_auc = result["auc"]
    promoted = result["promoted"]

    if promoted:
        msg = f"New model (AUC {new_auc:.4f}) promoted to Production."
    else:
        msg = (
            f"Existing champion retained (champion AUC {champion:.4f}, new AUC {new_auc:.4f}). "
            f"Improvement {new_auc - champion:.4f} did not exceed threshold 0.005."
        )

    return RetrainResponse(
        status="ok",
        new_auc=round(new_auc, 4),
        champion_auc=round(champion, 4) if champion is not None else None,
        promoted=promoted,
        message=msg,
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
def model_info():
    """Active Production model version, AUC, F1, and training date from MLflow."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if not versions:
            raise HTTPException(status_code=404, detail="No Production model registered.")
        v = versions[0]
        run = client.get_run(v.run_id)
        ts = run.info.start_time
        training_date = (
            datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc)
            .strftime("%Y-%m-%d %H:%M UTC")
            if ts
            else None
        )
        return ModelInfo(
            model_name=MODEL_NAME,
            version=str(v.version),
            stage=MODEL_STAGE,
            auc=run.data.metrics.get("auc"),
            f1=run.data.metrics.get("f1"),
            training_date=training_date,
            model_type=run.data.params.get("model_type"),
            run_id=v.run_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/runs", response_model=list[RunInfo], tags=["Model"])
def model_runs(n: int = Query(default=10, ge=1, le=100, description="Number of recent runs to return")):
    """Last N MLflow experiment runs with AUC, F1, and timestamp."""
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    try:
        experiment = client.get_experiment_by_name(
            os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")
        )
        if experiment is None:
            return []
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=n,
        )
        result = []
        for r in runs:
            ts = r.info.start_time
            timestamp = (
                datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc)
                .strftime("%Y-%m-%d %H:%M UTC")
                if ts
                else None
            )
            result.append(
                RunInfo(
                    run_id=r.info.run_id,
                    model_type=r.data.params.get("model_type"),
                    auc=r.data.metrics.get("auc"),
                    f1=r.data.metrics.get("f1"),
                    timestamp=timestamp,
                    status=r.info.status,
                )
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Liveness check — confirms model is loaded and returns uptime."""
    pipeline = _state.get("pipeline")
    return HealthResponse(
        status="ok" if pipeline is not None else "degraded",
        model_loaded=pipeline is not None,
        model_name=MODEL_NAME if pipeline is not None else None,
        model_version=_state.get("model_version"),
        uptime_seconds=round(time.time() - _state["start_time"], 1),
    )


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        reload=True,
    )
