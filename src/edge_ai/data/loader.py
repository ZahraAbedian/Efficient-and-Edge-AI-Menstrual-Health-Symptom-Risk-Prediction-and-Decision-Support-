from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


KEY = ["id", "study_interval", "day_in_study"]

SYMPTOM_COLS = [
    "headaches", "cramps", "sorebreasts", "fatigue", "sleepissue",
    "moodswing", "stress", "foodcravings", "indigestion", "bloating",
]

SYMPTOM_MAP = {
    "Not at all": 0, "Very Low/Little": 1, "Low": 2,
    "Moderate": 3, "High": 4, "Very High": 5,
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
}


def load_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    path = data_dir / filename
    df = pd.read_csv(path)
    print(f"{filename}: shape={df.shape}")
    return df


def load_all_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "hs": load_csv(data_dir, "hormones_and_selfreport.csv"),
        "sleep": load_csv(data_dir, "sleep_score.csv"),
        "am": load_csv(data_dir, "active_minutes.csv"),
        "rhr": load_csv(data_dir, "resting_heart_rate.csv"),
        "hrv": load_csv(data_dir, "heart_rate_variability_details.csv"),
        "wt": load_csv(data_dir, "wrist_temperature.csv"),
    }


def engineer_target(hs: pd.DataFrame) -> pd.DataFrame:
    hs_symptoms = hs.copy()
    for col in SYMPTOM_COLS:
        hs_symptoms[col + "_num"] = hs_symptoms[col].map(SYMPTOM_MAP)

    symptom_num_cols = [col + "_num" for col in SYMPTOM_COLS]
    hs_symptoms["symptom_score"] = hs_symptoms[symptom_num_cols].sum(axis=1)

    threshold_75 = hs_symptoms["symptom_score"].quantile(0.75)
    hs_symptoms["high_symptom_day"] = (
        hs_symptoms["symptom_score"] >= threshold_75
    ).astype(int)

    hs_symptoms = hs_symptoms.sort_values(KEY)

    hs_symptoms["tomorrow_high_symptom"] = (
        hs_symptoms
        .groupby(["id", "study_interval"])["high_symptom_day"]
        .shift(-1)
    )

    target = hs_symptoms.dropna(subset=["tomorrow_high_symptom"]).copy()
    target["tomorrow_high_symptom"] = target["tomorrow_high_symptom"].astype(int)

    target = target[[*KEY, "is_weekend", "phase", "symptom_score",
                     "high_symptom_day", "tomorrow_high_symptom"]].copy()
    return target


def aggregate_rhr(rhr: pd.DataFrame) -> pd.DataFrame:
    rhr_daily = (
        rhr
        .groupby(KEY)
        .agg(
            rhr_mean=("value", "mean"),
            rhr_min=("value", "min"),
            rhr_max=("value", "max"),
            rhr_std=("value", "std"),
            rhr_error_mean=("error", "mean"),
        )
        .reset_index()
    )
    rhr_daily["rhr_std"] = rhr_daily["rhr_std"].fillna(0)
    return rhr_daily


def aggregate_hrv(hrv: pd.DataFrame) -> pd.DataFrame:
    hrv_daily = (
        hrv
        .groupby(KEY)
        .agg(
            hrv_rmssd_mean=("rmssd", "mean"),
            hrv_rmssd_min=("rmssd", "min"),
            hrv_rmssd_max=("rmssd", "max"),
            hrv_rmssd_std=("rmssd", "std"),
            hrv_coverage_mean=("coverage", "mean"),
            hrv_lf_mean=("low_frequency", "mean"),
            hrv_hf_mean=("high_frequency", "mean"),
        )
        .reset_index()
    )
    hrv_daily["hrv_rmssd_std"] = hrv_daily["hrv_rmssd_std"].fillna(0)
    return hrv_daily


def aggregate_wrist_temp(wt: pd.DataFrame) -> pd.DataFrame:
    wt_daily = (
        wt
        .groupby(KEY)
        .agg(
            temp_mean=("temperature_diff_from_baseline", "mean"),
            temp_min=("temperature_diff_from_baseline", "min"),
            temp_max=("temperature_diff_from_baseline", "max"),
            temp_std=("temperature_diff_from_baseline", "std"),
        )
        .reset_index()
    )
    wt_daily["temp_std"] = wt_daily["temp_std"].fillna(0)
    return wt_daily
