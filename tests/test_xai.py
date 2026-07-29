import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from edge_ai.xai.explainer import Explainer


def test_explainer_coefficient_fallback():
    model = Pipeline([
        ("preprocessor", "passthrough"),
        ("classifier", LogisticRegression(random_state=42)),
    ])
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    y = np.array([0, 1])
    model.fit(X, y)

    explainer = Explainer(model, feature_names=["a", "b"], background_df=X, threshold=0.5)
    result = explainer.explain(X.iloc[[0]])

    assert "probability" in result
    assert "prediction" in result
    assert "risk_level" in result
    assert "explanation" in result

    explanation = result["explanation"]
    assert explanation["method"] in ("coefficients", "shap")


def test_risk_level():
    model = Pipeline([
        ("preprocessor", "passthrough"),
        ("classifier", LogisticRegression(random_state=42)),
    ])
    X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    y = np.array([0, 1])
    model.fit(X, y)
    explainer = Explainer(model, feature_names=["a", "b"], background_df=X, threshold=0.35)
    assert explainer._risk_level(0.2) == "Low"
    assert explainer._risk_level(0.35) == "Medium"
    assert explainer._risk_level(0.7) == "High"


def test_make_planning_card():
    explanation = {
        "method": "coefficients",
        "top_features": [
            {"feature": "resting_heart_rate", "coefficient": 0.5,
             "impact_direction": "increases_risk"},
        ],
    }
    card = Explainer.make_planning_card(explanation)
    assert "resting_heart_rate" in card
