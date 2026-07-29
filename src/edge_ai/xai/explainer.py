from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

SYMPTOM_COLS = [
    "headaches", "cramps", "sorebreasts", "fatigue", "sleepissue",
    "moodswing", "stress", "foodcravings", "indigestion", "bloating",
]


class Explainer:
    def __init__(
        self,
        model: object,
        feature_names: list[str],
        background_df: Optional[pd.DataFrame] = None,
        threshold: float = 0.5,
    ):
        self.model = model
        self.feature_names = feature_names
        self.background_df = background_df
        self.threshold = threshold
        self._shap_explainer = None
        self._coefficients = None
        self._has_shap = False

        self._setup()

    def _setup(self):
        try:
            import shap
            if self.background_df is not None and len(self.background_df) > 1:
                classifier = self.model.named_steps.get("classifier")
                if classifier is not None:
                    self._shap_explainer = shap.LinearExplainer(
                        classifier,
                        self.background_df,
                        feature_perturbation="correlation_dependent",
                    )
                    self._has_shap = True
        except Exception:
            pass

        if not self._has_shap:
            try:
                self._coefficients = self._extract_coefficients()
            except Exception:
                self._coefficients = None

    def _extract_coefficients(self) -> dict[str, float]:
        classifier = self.model.named_steps.get("classifier")
        if classifier is None or not hasattr(classifier, "coef_"):
            return {}

        preprocessor = self.model.named_steps.get("preprocessor")
        coef = classifier.coef_[0]

        try:
            if preprocessor is not None and preprocessor != "passthrough":
                if hasattr(preprocessor, "get_feature_names_out"):
                    feature_names_out = preprocessor.get_feature_names_out()
                else:
                    feature_names_out = self.feature_names
            else:
                feature_names_out = self.feature_names
        except Exception:
            feature_names_out = self.feature_names

        if len(coef) != len(feature_names_out):
            feature_names_out = [f"feature_{i}" for i in range(len(coef))]

        return dict(zip(feature_names_out, coef))

    def explain(self, sample: pd.DataFrame) -> dict:
        shap_values = None
        if self._has_shap and self._shap_explainer is not None:
            try:
                preprocessor = self.model.named_steps.get("preprocessor")
                if preprocessor is not None and preprocessor != "passthrough":
                    transformed = preprocessor.transform(sample)
                else:
                    transformed = sample
                shap_values = self._shap_explainer.shap_values(transformed)
            except Exception:
                shap_values = None

        prob = float(self.model.predict_proba(sample)[:, 1][0])
        pred = int(prob >= self.threshold)
        risk = self._risk_level(prob)

        if shap_values is not None and len(shap_values) > 0:
            explanation = self._format_shap(shap_values[0], prob)
        elif self._coefficients:
            explanation = self._format_coefficients()
        else:
            explanation = {"method": "none", "detail": "No explanation available"}

        return {
            "probability": prob,
            "prediction": pred,
            "risk_level": risk,
            "explanation": explanation,
        }

    def _format_shap(self, shap_values: np.ndarray, prob: float) -> dict:
        preprocessor = self.model.named_steps.get("preprocessor")
        try:
            if preprocessor is not None and preprocessor != "passthrough" and hasattr(preprocessor, "get_feature_names_out"):
                feature_names_out = list(preprocessor.get_feature_names_out())
            else:
                feature_names_out = self.feature_names
        except Exception:
            feature_names_out = [f"feature_{i}" for i in range(len(shap_values))]

        if len(shap_values) != len(feature_names_out):
            feature_names_out = feature_names_out[:len(shap_values)]

        abs_vals = np.abs(shap_values)
        top_indices = np.argsort(abs_vals)[::-1][:5]

        top_features = []
        for idx in top_indices:
            top_features.append({
                "feature": feature_names_out[idx],
                "shap_value": float(round(shap_values[idx], 4)),
                "impact_direction": "increases_risk" if shap_values[idx] > 0 else "decreases_risk",
            })

        return {
            "method": "shap",
            "top_features": top_features,
            "base_value": float(prob),
        }

    def _format_coefficients(self) -> dict:
        if not self._coefficients:
            return {"method": "none", "detail": "No coefficients available"}

        sorted_coefs = sorted(
            self._coefficients.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]

        top_features = []
        for feature, coef in sorted_coefs:
            top_features.append({
                "feature": feature,
                "coefficient": float(round(coef, 4)),
                "impact_direction": "increases_risk" if coef > 0 else "decreases_risk",
            })

        return {
            "method": "coefficients",
            "top_features": top_features,
        }

    def _risk_level(self, prob: float) -> str:
        if prob < self.threshold - 0.1:
            return "Low"
        elif prob > self.threshold + 0.1:
            return "High"
        else:
            return "Medium"

    @staticmethod
    def make_planning_card(explanation: dict) -> str:
        if explanation.get("method") == "none":
            return "No explanation available for planning."

        top = explanation.get("top_features", [])
        if not top:
            return "No contributing factors found."

        risk_word = "higher" if top[0]["impact_direction"] == "increases_risk" else "lower"
        main_factor = top[0]["feature"]

        lines = [f"Your risk is influenced most by **{main_factor}**."]
        lines.append(
            f"A {risk_word} value of this factor suggests you may need to take extra care today."
        )
        lines.append("")
        lines.append("Suggested actions:")

        if risk_word == "higher":
            lines.append("- Prioritise rest and self-care today.")
            lines.append("- Monitor how you feel and plan lighter activities.")
        else:
            lines.append("- Your current readings look favourable.")
            lines.append("- Maintain your usual routine while staying mindful.")

        if len(top) > 1:
            secondary = top[1]["feature"]
            lines.append(
                f"- Also keep an eye on **{secondary}**, which is another contributing factor."
            )

        return "\n".join(lines)


class ModelManager:
    MODEL_DIR = Path.home() / "edge_ai_models"
    FILES = {
        "pipeline": "final_edge_ai_symptom_risk_pipeline.joblib",
        "features": "final_edge_ai_input_feature_names.joblib",
        "threshold": "final_edge_ai_threshold.joblib",
        "sample_input": "edge_sample_input.csv",
    }

    def __init__(self, model_dir: str | Path = MODEL_DIR):
        d = Path(model_dir).expanduser()
        pipeline = joblib.load(d / self.FILES["pipeline"])
        feature_names = joblib.load(d / self.FILES["features"])
        threshold = joblib.load(d / self.FILES["threshold"])
        sample = pd.read_csv(d / self.FILES["sample_input"])
        sample = sample[feature_names]

        self._latest_input = sample
        self._explainer = Explainer(
            model=pipeline,
            feature_names=feature_names,
            background_df=sample,
            threshold=threshold,
        )

    def get_latest_input(self) -> pd.DataFrame:
        return self._latest_input

    def explain(self, sample: pd.DataFrame) -> dict:
        return self._explainer.explain(sample)

    @staticmethod
    def make_planning_card(explanation: dict) -> str:
        return Explainer.make_planning_card(explanation)
