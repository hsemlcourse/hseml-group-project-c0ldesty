import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Собирает два варианта preprocessing: для линейных моделей и для деревьев.
def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    linear_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return linear_preprocessor, tree_preprocessor, numeric_features, categorical_features


# Считает основные метрики по предсказаниям; используется и в cp2 для сравнения моделей по score.
def get_scores_from_predictions(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = float(np.nan)
        metrics["pr_auc"] = float(np.nan)
    return metrics


# Обучает одну модель и возвращает и саму модель, и метрики на validation; используется в cp2.
def fit_and_score(model, X_fit: pd.DataFrame, y_fit: pd.Series, X_eval: pd.DataFrame, y_eval: pd.Series):
    fitted_model = clone(model)
    fitted_model.fit(X_fit, y_fit)

    y_pred = fitted_model.predict(X_eval)

    y_score = None
    if hasattr(fitted_model, "predict_proba"):
        y_score = fitted_model.predict_proba(X_eval)[:, 1]
    elif hasattr(fitted_model, "decision_function"):
        y_score = fitted_model.decision_function(X_eval)

    scores = get_scores_from_predictions(y_eval, y_pred, y_score)
    return fitted_model, scores


# Запускает одинаковую оценку для набора моделей и собирает результаты в таблицу; используется в cp2.
def evaluate_model_dict(
    model_dict: dict[str, object],
    X_train_part: pd.DataFrame,
    y_train_part: pd.Series,
    X_val_part: pd.DataFrame,
    y_val_part: pd.Series,
    stage_name: str,
    feature_set_name: str,
):
    rows = []
    fitted = {}

    for model_name, model in model_dict.items():
        fitted_model, scores = fit_and_score(model, X_train_part, y_train_part, X_val_part, y_val_part)
        fitted[(feature_set_name, stage_name, model_name)] = fitted_model

        row = {
            "feature_set": feature_set_name,
            "stage": stage_name,
            "model": model_name,
        }
        row.update(scores)
        rows.append(row)

    return pd.DataFrame(rows), fitted
