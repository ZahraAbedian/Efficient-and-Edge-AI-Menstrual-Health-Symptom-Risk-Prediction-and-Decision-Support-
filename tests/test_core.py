import pandas as pd
import numpy as np

from edge_ai.data.loader import (
    SYMPTOM_MAP, SYMPTOM_COLS, KEY,
    engineer_target, aggregate_rhr, aggregate_hrv, aggregate_wrist_temp,
)
from edge_ai.features.builders import (
    V1_FEATURE_COLS, build_v1, build_v2, build_v3, build_v4,
)
from edge_ai.models.pipeline import (
    build_logistic_regression_pipeline,
    cross_validate_model, summarize_scores, find_best_threshold,
)


def test_symptom_map():
    assert SYMPTOM_MAP["Not at all"] == 0
    assert SYMPTOM_MAP["Very High"] == 5
    assert SYMPTOM_MAP["1"] == 1


def test_key_columns():
    assert KEY == ["id", "study_interval", "day_in_study"]


def test_engineer_target():
    hs = pd.DataFrame({
        "id": [1, 1],
        "study_interval": [2022, 2022],
        "is_weekend": [True, False],
        "day_in_study": [1, 2],
        "phase": ["Follicular", "Follicular"],
        **{col: ["Not at all"] * 2 for col in SYMPTOM_COLS},
    })
    target = engineer_target(hs)
    assert "tomorrow_high_symptom" in target.columns
    assert "phase" in target.columns
    assert len(target) > 0


def test_aggregate_rhr():
    rhr = pd.DataFrame({
        "id": [1, 1],
        "study_interval": [2022, 2022],
        "is_weekend": [True, False],
        "day_in_study": [1, 1],
        "value": [70.0, 72.0],
        "error": [5.0, 3.0],
    })
    daily = aggregate_rhr(rhr)
    assert "rhr_mean" in daily.columns
    assert daily["rhr_mean"].iloc[0] == 71.0


def test_aggregate_hrv():
    hrv = pd.DataFrame({
        "id": [1, 1],
        "study_interval": [2022, 2022],
        "is_weekend": [True, False],
        "day_in_study": [1, 1],
        "rmssd": [30.0, 40.0],
        "coverage": [0.8, 0.9],
        "low_frequency": [0.3, 0.4],
        "high_frequency": [0.7, 0.6],
    })
    daily = aggregate_hrv(hrv)
    assert "hrv_rmssd_mean" in daily.columns
    assert daily["hrv_rmssd_mean"].iloc[0] == 35.0


def test_aggregate_wrist_temp():
    wt = pd.DataFrame({
        "id": [1, 1],
        "study_interval": [2022, 2022],
        "is_weekend": [True, False],
        "day_in_study": [1, 1],
        "temperature_diff_from_baseline": [0.5, 1.5],
    })
    daily = aggregate_wrist_temp(wt)
    assert "temp_mean" in daily.columns
    assert daily["temp_mean"].iloc[0] == 1.0


def test_v1_feature_cols():
    assert "is_weekend" in V1_FEATURE_COLS
    assert "phase" in V1_FEATURE_COLS
    assert "overall_score" in V1_FEATURE_COLS


def test_build_v1():
    target = pd.DataFrame({**{k: [1] for k in KEY}, "is_weekend": [True],
                           "phase": ["Follicular"], "tomorrow_high_symptom": [0]})
    sleep = pd.DataFrame({**{k: [1] for k in KEY}, "overall_score": [80.0],
                          "revitalization_score": [50.0], "deep_sleep_in_minutes": [90.0],
                          "resting_heart_rate": [70.0], "restlessness": [0.1]})
    am = pd.DataFrame({**{k: [1] for k in KEY}, "lightly": [60.0],
                        "moderately": [10.0], "very": [5.0]})
    df = build_v1(target, sleep, am)
    assert "tomorrow_high_symptom" in df.columns
    assert df["overall_score"].iloc[0] == 80.0


def test_pipeline_creation():
    pipe = build_logistic_regression_pipeline(
        numeric_features=["overall_score", "deep_sleep_in_minutes"],
        categorical_features=["phase"],
    )
    assert pipe is not None
    assert "preprocessor" in pipe.named_steps
    assert "classifier" in pipe.named_steps


def test_find_best_threshold():
    y_true = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.3, 0.4, 0.2, 0.9, 0.3])
    threshold, metrics = find_best_threshold(y_true, y_prob, thresholds=[0.3, 0.5, 0.7])
    assert 0.3 <= threshold <= 0.7
    assert "f1" in metrics
    assert "recall" in metrics
