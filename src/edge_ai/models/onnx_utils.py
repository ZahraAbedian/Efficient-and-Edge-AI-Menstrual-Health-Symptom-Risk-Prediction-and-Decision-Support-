from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def convert_to_onnx(
    pipeline: Pipeline,
    output_path: str | Path,
    sample_input: pd.DataFrame,
) -> None:
    try:
        from skl2onnx import to_onnx
    except ImportError:
        raise ImportError(
            "skl2onnx is required for ONNX conversion. "
            "Install it with: pip install skl2onnx"
        )
    pipe = _make_onnx_compatible(pipeline)

    options = {id(pipe): {"zipmap": False}}
    try:
        onnx_model = to_onnx(
            pipe, sample_input.astype(np.float32, errors="ignore"),
            options=options,
        )
    except Exception:
        onnx_model = to_onnx(
            pipe, sample_input.astype(np.float32, errors="ignore"),
        )
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def _make_onnx_compatible(pipeline: Pipeline) -> Pipeline:
    import copy
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SkPipe
    from sklearn.compose import ColumnTransformer as ColT

    pipe = copy.deepcopy(pipeline)
    preprocessor = pipe.named_steps.get("preprocessor")
    if preprocessor is None or not isinstance(preprocessor, ColT):
        return pipe

    transformers = list(preprocessor.transformers)
    for i, (name, transformer, columns) in enumerate(transformers):
        if isinstance(transformer, SkPipe):
            steps = list(transformer.steps)
            for j, (step_name, step_obj) in enumerate(steps):
                if isinstance(step_obj, SimpleImputer):
                    kwargs = {"strategy": "constant", "fill_value": "missing"}
                    try:
                        if step_obj.statistics_ is not None and len(step_obj.statistics_) > 0:
                            pass
                    except Exception:
                        pass
                    replacement = SimpleImputer(**kwargs)
                    if hasattr(step_obj, "statistics_"):
                        replacement.statistics_ = step_obj.statistics_
                    steps[j] = (step_name, replacement)
            transformer = SkPipe(steps)
            transformers[i] = (name, transformer, columns)

    preprocessor.transformers = transformers
    return pipe


def validate_onnx(
    onnx_path: str | Path,
    pipeline: Pipeline,
    sample_df: pd.DataFrame,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for ONNX validation. "
            "Install it with: pip install onnxruntime"
        )

    session = ort.InferenceSession(str(onnx_path))
    onnx_inputs = {}
    for inp in session.get_inputs():
        col = inp.name
        if col in sample_df.columns:
            onnx_inputs[col] = sample_df[[col]].astype(np.float32, errors="ignore").to_numpy()
        else:
            idx = int(col.replace("float_input_", ""))
            onnx_inputs[col] = sample_df.iloc[:, [idx]].astype(np.float32, errors="ignore").to_numpy()

    onnx_out = session.run(None, onnx_inputs)
    onnx_probs = onnx_out[-1] if len(onnx_out) > 1 else onnx_out[0]

    sklearn_probs = pipeline.predict_proba(sample_df)

    if sklearn_probs.shape[1] == 2:
        sklearn_probs_class1 = sklearn_probs[:, 1]
    else:
        sklearn_probs_class1 = sklearn_probs[:, 0]

    if onnx_probs.ndim == 2 and onnx_probs.shape[1] == 2:
        onnx_probs_class1 = onnx_probs[:, 1]
    elif onnx_probs.ndim == 2 and onnx_probs.shape[1] == 1:
        onnx_probs_class1 = onnx_probs[:, 0]
    else:
        onnx_probs_class1 = onnx_probs

    max_diff = float(np.max(np.abs(sklearn_probs_class1 - onnx_probs_class1)))
    mean_diff = float(np.mean(np.abs(sklearn_probs_class1 - onnx_probs_class1)))
    passed = max_diff < max(rtol, atol)

    return {
        "max_abs_difference": max_diff,
        "mean_abs_difference": mean_diff,
        "passed": passed,
        "sklearn_shape": sklearn_probs.shape,
        "onnx_shape": onnx_probs.shape,
    }


def _make_onnx_input(session: "onnxruntime.InferenceSession", row: pd.DataFrame) -> dict:
    onnx_inputs = {}
    for inp in session.get_inputs():
        col = inp.name
        if col in row.columns:
            onnx_inputs[col] = row[[col]].astype(np.float32, errors="ignore").to_numpy()
        else:
            idx = int(col.replace("float_input_", ""))
            onnx_inputs[col] = row.iloc[:, [idx]].astype(np.float32, errors="ignore").to_numpy()
    return onnx_inputs


def benchmark_latency(
    pipeline: Pipeline,
    session: "onnxruntime.InferenceSession",
    sample: pd.DataFrame,
    n_runs: int = 1000,
) -> dict:
    import time

    sklearn_latencies = []
    onnx_latencies = []

    pipeline.predict_proba(sample.iloc[[0]])
    session.run(None, _make_onnx_input(session, sample.iloc[[0]]))

    for _ in range(n_runs):
        row = sample.sample(1)

        t0 = time.perf_counter()
        pipeline.predict_proba(row)
        t1 = time.perf_counter()
        sklearn_latencies.append((t1 - t0) * 1000)

        t0 = time.perf_counter()
        session.run(None, _make_onnx_input(session, row))
        t1 = time.perf_counter()
        onnx_latencies.append((t1 - t0) * 1000)

    def summarize(arr):
        arr = np.array(arr)
        return {
            "mean_ms": float(round(arr.mean(), 4)),
            "median_ms": float(round(np.median(arr), 4)),
            "std_ms": float(round(arr.std(), 4)),
            "p95_ms": float(round(np.percentile(arr, 95), 4)),
            "p99_ms": float(round(np.percentile(arr, 99), 4)),
            "min_ms": float(round(arr.min(), 4)),
            "max_ms": float(round(arr.max(), 4)),
        }

    return {
        "sklearn": summarize(sklearn_latencies),
        "onnx": summarize(onnx_latencies),
    }
