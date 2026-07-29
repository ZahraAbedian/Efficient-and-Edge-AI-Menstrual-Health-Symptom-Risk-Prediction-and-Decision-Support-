import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, make_scorer,
)

RANDOM_STATE = 42

SCORERS = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score),
    "f1": make_scorer(f1_score),
    "pr_auc": make_scorer(average_precision_score),
}


def build_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def build_logistic_regression_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    **lr_kwargs,
) -> Pipeline:
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        **lr_kwargs,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", lr)])


def cross_validate_model(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
) -> dict[str, np.ndarray]:
    cv = GroupKFold(n_splits=n_splits)
    scores = cross_validate(
        pipeline, X, y,
        cv=cv, groups=groups,
        scoring=SCORERS,
        return_train_score=False,
    )
    return scores


def summarize_scores(scores: dict[str, np.ndarray]) -> dict[str, float]:
    summary = {}
    for metric in ["accuracy", "precision", "recall", "f1", "pr_auc"]:
        test_key = f"test_{metric}"
        if test_key in scores:
            summary[metric] = float(scores[test_key].mean())
            summary[f"{metric}_std"] = float(scores[test_key].std())
    return summary


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.71, 0.05)]
    best_f1 = -1
    best_threshold = 0.5
    best_metrics = {}
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics = {
                "threshold": t,
                "f1": f1,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred),
            }
    return best_threshold, best_metrics
