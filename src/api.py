from functools import lru_cache
from typing import Any

import pandas as pd
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from .modeling import build_preprocessor
from .preprocessing import PROCESSED_DATA_DIR, add_feature_engineering


TARGET_COLUMN = "FraudFound_P"
FINAL_THRESHOLD = 0.33
PROCESSED_FE_PATH = PROCESSED_DATA_DIR / "fraud_oracle_fe.csv"
RANDOM_STATE = 42


class PredictionItem(BaseModel):
    fraud_probability: float
    fraud_prediction: int


class PredictionResponse(BaseModel):
    threshold: float
    predictions: list[PredictionItem]


class ModelBundle:
    def __init__(self, pipeline: Pipeline, feature_columns: list[str]) -> None:
        self.pipeline = pipeline
        self.feature_columns = feature_columns


app = FastAPI(
    title="Vehicle Insurance Fraud Detection API",
    description="API for fraud prediction on vehicle insurance claims.",
    version="1.0.0",
)


def _build_final_pipeline(X: pd.DataFrame) -> Pipeline:
    _, tree_preprocessor, _, _ = build_preprocessor(X)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=400,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5,
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=0,
    )
    return Pipeline(
        [
            ("preprocessor", tree_preprocessor),
            ("model", model),
        ]
    )


@lru_cache(maxsize=1)
def get_model_bundle() -> ModelBundle:
    if not PROCESSED_FE_PATH.exists():
        raise FileNotFoundError(
            f"Processed feature dataset was not found: {PROCESSED_FE_PATH}. "
            "Run notebooks/01_eda.ipynb or src.preprocessing.prepare_datasets first."
        )

    data = pd.read_csv(PROCESSED_FE_PATH)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column is missing from processed dataset: {TARGET_COLUMN}")

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    pipeline = _build_final_pipeline(X)
    pipeline.fit(X, y)

    return ModelBundle(pipeline=pipeline, feature_columns=X.columns.tolist())


def _normalize_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "records" in payload:
        records = payload["records"]
        if not isinstance(records, list) or not records or not all(isinstance(item, dict) for item in records):
            raise HTTPException(
                status_code=422,
                detail="records must be a non-empty list of JSON objects.",
            )
        return records
    return [payload]


def _prepare_request_frame(records: list[dict[str, Any]], feature_columns: list[str]) -> pd.DataFrame:
    request_df = pd.DataFrame(records)
    request_df = request_df.drop(columns=[TARGET_COLUMN, "PolicyNumber", "RepNumber"], errors="ignore")

    if "Age" in request_df.columns:
        request_df["Age"] = request_df["Age"].replace({0: 16.5})

    missing_engineered = [col for col in feature_columns if col not in request_df.columns]
    if missing_engineered:
        try:
            request_df = add_feature_engineering(request_df)
        except KeyError:
            pass

    missing_columns = [col for col in feature_columns if col not in request_df.columns]
    if missing_columns:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Request is missing required feature columns.",
                "missing_columns": missing_columns,
            },
        )

    return request_df[feature_columns]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    bundle = get_model_bundle()
    return {
        "model": "CatBoostClassifier",
        "target": TARGET_COLUMN,
        "threshold": FINAL_THRESHOLD,
        "feature_count": len(bundle.feature_columns),
        "processed_dataset": str(PROCESSED_FE_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: dict[str, Any]) -> PredictionResponse:
    try:
        bundle = get_model_bundle()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    records = _normalize_records(payload)
    X_request = _prepare_request_frame(records, bundle.feature_columns)
    probabilities = bundle.pipeline.predict_proba(X_request)[:, 1]

    predictions = [
        {
            "fraud_probability": float(probability),
            "fraud_prediction": int(probability >= FINAL_THRESHOLD),
        }
        for probability in probabilities
    ]

    return PredictionResponse(threshold=FINAL_THRESHOLD, predictions=predictions)
