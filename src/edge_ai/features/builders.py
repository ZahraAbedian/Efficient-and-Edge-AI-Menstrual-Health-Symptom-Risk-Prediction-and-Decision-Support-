import pandas as pd

from edge_ai.data.loader import KEY


V1_COLS = [
    *KEY, "is_weekend", "phase",
    "overall_score", "revitalization_score",
    "deep_sleep_in_minutes", "resting_heart_rate", "restlessness",
    "lightly", "moderately", "very",
    "tomorrow_high_symptom",
]

V1_FEATURE_COLS = [
    "is_weekend", "phase",
    "overall_score", "revitalization_score",
    "deep_sleep_in_minutes", "resting_heart_rate", "restlessness",
    "lightly", "moderately", "very",
]


def build_v1(target: pd.DataFrame, sleep: pd.DataFrame, am: pd.DataFrame) -> pd.DataFrame:
    merged = target.merge(sleep, on=KEY, how="left", suffixes=("", "_sleep"))
    merged = merged.merge(am, on=KEY, how="left", suffixes=("", "_activity"))
    return merged[V1_COLS].copy()


def build_v2(
    target: pd.DataFrame, sleep: pd.DataFrame,
    am: pd.DataFrame, rhr_daily: pd.DataFrame,
) -> pd.DataFrame:
    merged = target.merge(sleep, on=KEY, how="left", suffixes=("", "_sleep"))
    merged = merged.merge(am, on=KEY, how="left", suffixes=("", "_activity"))
    merged = merged.merge(rhr_daily, on=KEY, how="left")
    v2_cols = V1_FEATURE_COLS + [
        "rhr_mean", "rhr_min", "rhr_max", "rhr_std", "rhr_error_mean",
        "tomorrow_high_symptom",
    ]
    return merged[[*KEY, *v2_cols]].copy()


def build_v3(
    target: pd.DataFrame, sleep: pd.DataFrame,
    am: pd.DataFrame, rhr_daily: pd.DataFrame,
    hrv_daily: pd.DataFrame,
) -> pd.DataFrame:
    merged = target.merge(sleep, on=KEY, how="left", suffixes=("", "_sleep"))
    merged = merged.merge(am, on=KEY, how="left", suffixes=("", "_activity"))
    merged = merged.merge(rhr_daily, on=KEY, how="left")
    merged = merged.merge(hrv_daily, on=KEY, how="left")
    v3_cols = V1_FEATURE_COLS + [
        "rhr_mean", "rhr_min", "rhr_max", "rhr_std", "rhr_error_mean",
        "hrv_rmssd_mean", "hrv_rmssd_min", "hrv_rmssd_max", "hrv_rmssd_std",
        "hrv_coverage_mean", "hrv_lf_mean", "hrv_hf_mean",
        "tomorrow_high_symptom",
    ]
    return merged[[*KEY, *v3_cols]].copy()


def build_v4(
    target: pd.DataFrame, sleep: pd.DataFrame,
    am: pd.DataFrame, rhr_daily: pd.DataFrame,
    hrv_daily: pd.DataFrame, wt_daily: pd.DataFrame,
) -> pd.DataFrame:
    merged = target.merge(sleep, on=KEY, how="left", suffixes=("", "_sleep"))
    merged = merged.merge(am, on=KEY, how="left", suffixes=("", "_activity"))
    merged = merged.merge(rhr_daily, on=KEY, how="left")
    merged = merged.merge(hrv_daily, on=KEY, how="left")
    merged = merged.merge(wt_daily, on=KEY, how="left")
    v4_cols = V1_FEATURE_COLS + [
        "rhr_mean", "rhr_min", "rhr_max", "rhr_std", "rhr_error_mean",
        "hrv_rmssd_mean", "hrv_rmssd_min", "hrv_rmssd_max", "hrv_rmssd_std",
        "hrv_coverage_mean", "hrv_lf_mean", "hrv_hf_mean",
        "temp_mean", "temp_min", "temp_max", "temp_std",
        "tomorrow_high_symptom",
    ]
    return merged[[*KEY, *v4_cols]].copy()
