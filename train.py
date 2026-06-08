"""
train.py — Load data, train LogisticRegression pipeline, log to MLflow, register model.
Run directly: python train.py
"""

import os
import warnings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "churn-prediction")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "ChurnModel")
DATA_PATH = os.getenv(
    "DATA_PATH",
    str(BASE_DIR / "Dataset" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
)

# Logistic Regression hyperparams
LR_C = float(os.getenv("LR_C", "1.0"))
LR_MAX_ITER = int(os.getenv("LR_MAX_ITER", "1000"))
PROMOTE_THRESHOLD = float(os.getenv("PROMOTE_THRESHOLD", "0.005"))

# ── Feature definitions ───────────────────────────────────────────────────────
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
TARGET = "Churn"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges has spaces for new customers — convert to float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def build_pipeline(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    clf = LogisticRegression(
        class_weight="balanced",
        C=c,
        max_iter=max_iter,
        random_state=42,
        solver="lbfgs",
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])


def get_champion_auc(client: MlflowClient, model_name: str) -> float | None:
    """Return AUC of the current Production model, or None if none exists."""
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None
        run = client.get_run(versions[0].run_id)
        return run.data.metrics.get("auc")
    except Exception:
        return None


def register_and_promote(
    client: MlflowClient,
    run_id: str,
    model_name: str,
    new_auc: float,
    champion_auc: float | None,
) -> bool:
    """Register model version, promote to Production if it beats the champion."""
    mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client.transition_model_version_stage(
        name=model_name, version=mv.version, stage="Staging"
    )
    promoted = False
    if champion_auc is None or (new_auc - champion_auc) > PROMOTE_THRESHOLD:
        client.transition_model_version_stage(
            name=model_name, version=mv.version, stage="Production"
        )
        promoted = True
        print(f"  Promoted version {mv.version} to Production (AUC {new_auc:.4f})")
    else:
        print(
            f"  Kept existing champion (champion AUC {champion_auc:.4f} vs new {new_auc:.4f})"
        )
    return promoted


def train(data_path: str = DATA_PATH) -> dict:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    print(f"Loading data from {data_path} ...")
    df = load_data(data_path)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  Churn rate: {y.mean():.1%}")

    champion_auc = get_champion_auc(client, MODEL_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow run: {run_id}")

        # Params
        mlflow.log_params(
            {
                "model_type": "logistic_regression",
                "C": LR_C,
                "max_iter": LR_MAX_ITER,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "train_size": len(X_train),
                "test_size": len(X_test),
            }
        )

        # Train
        pipeline = build_pipeline(c=LR_C, max_iter=LR_MAX_ITER)
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metrics({"auc": auc, "f1": f1})
        print(f"  AUC: {auc:.4f}  F1: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

        # Log model artifact
        mlflow.sklearn.log_model(pipeline, "model")

    # Register and optionally promote
    promoted = register_and_promote(client, run_id, MODEL_NAME, auc, champion_auc)

    return {
        "run_id": run_id,
        "auc": auc,
        "f1": f1,
        "champion_auc": champion_auc,
        "promoted": promoted,
    }


if __name__ == "__main__":
    result = train()
    print("\nDone:", result)
