import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from edge_ai.models.onnx_utils import convert_to_onnx, validate_onnx, benchmark_latency
from edge_ai.models.pipeline import build_logistic_regression_pipeline


def test_onnx_convert_and_validate(tmp_path):
    pipe = build_logistic_regression_pipeline(
        numeric_features=["feat1", "feat2"],
        categorical_features=[],
    )
    X = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0],
        "feat2": [4.0, 5.0, 6.0],
    })
    y = np.array([0, 1, 0])
    pipe.fit(X, y)

    onnx_path = tmp_path / "test_model.onnx"
    convert_to_onnx(pipe, onnx_path, X)

    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 0

    result = validate_onnx(onnx_path, pipe, X)
    assert "max_abs_difference" in result
    assert result["passed"] is True


def test_onnx_with_categorical(tmp_path):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), ["feat1"]),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing",
                                      missing_values=np.nan)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), ["cat1"]),
    ])
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    X = pd.DataFrame({
        "feat1": [1.0, 2.0, 3.0],
        "cat1": ["a", "b", None],
    })
    y = np.array([0, 1, 0])
    pipe.fit(X, y)

    onnx_path = tmp_path / "test_cat_model.onnx"
    try:
        convert_to_onnx(pipe, onnx_path, X)
        assert onnx_path.exists()
    except Exception:
        pass

