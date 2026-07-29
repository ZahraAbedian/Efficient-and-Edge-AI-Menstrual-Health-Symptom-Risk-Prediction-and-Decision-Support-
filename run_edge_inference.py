#!/usr/bin/env python3
"""Edge AI on-device inference test for Raspberry Pi.

Measures latency (P50/P95/P99), power (vcgencmd), RAM/CPU,
and logs results to SQLite for the Gradio dashboard.

Usage:
    python run_edge_inference.py [--model-dir DIR] [--n-runs N]
"""

import argparse
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil

from edge_ai.monitoring.metrics import (
    setup_database,
    log_inference,
    measure_power,
)


def risk_level(prob: float, threshold: float) -> str:
    if prob < threshold - 0.1:
        return "Low"
    elif prob > threshold + 0.1:
        return "High"
    return "Medium"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edge AI on-device inference test for Raspberry Pi"
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory with model artifacts (default: script directory)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=1000,
        help="Number of latency test iterations (default: 1000)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "onnx", "joblib"],
        default="auto",
        help="Inference backend (default: auto-pick ONNX if available)",
    )
    return parser.parse_args()


def run_latency_test(predict_fn, sample, n_runs, n_warmup):
    for _ in range(n_warmup):
        predict_fn(sample)
    latencies = []
    for i in range(n_runs):
        row = sample.sample(1)
        t0 = time.perf_counter()
        predict_fn(row)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n_runs} iterations", flush=True)
    arr = np.array(latencies)
    return {
        "mean_ms": round(float(arr.mean()), 4),
        "median_ms": round(float(np.median(arr)), 4),
        "std_ms": round(float(arr.std()), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "min_ms": round(float(arr.min()), 4),
        "max_ms": round(float(arr.max()), 4),
    }


def main() -> None:
    args = parse_args()

    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser().resolve()
    else:
        model_dir = Path(__file__).parent.resolve()

    model_path = model_dir / "final_edge_ai_symptom_risk_pipeline.joblib"
    onnx_path = model_dir / "final_edge_ai_symptom_risk_pipeline.onnx"
    feature_list_path = model_dir / "final_edge_ai_input_feature_names.joblib"
    threshold_path = model_dir / "final_edge_ai_threshold.joblib"
    input_path = model_dir / "edge_sample_input.csv"
    results_path = model_dir / "raspberry_pi_edge_results.csv"

    db_path = setup_database()
    n_runs = args.n_runs
    n_warmup = 10

    def log(msg: str) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    def check(path: Path) -> None:
        if not path.exists():
            print(f"[ERROR] Missing: {path}", flush=True)
            sys.exit(1)

    log("Step 1/9: Checking files...")
    for p in [model_path, feature_list_path, threshold_path, input_path]:
        check(p)
    use_onnx = onnx_path.exists() and args.backend != "joblib"
    if use_onnx:
        log("  ONNX model found")

    log("Step 2/9: Loading model...")
    pipeline = joblib.load(model_path)
    feature_list = joblib.load(feature_list_path)
    threshold = float(joblib.load(threshold_path))

    if use_onnx:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))

        def _onnx_input(row):
            inp = {}
            for meta in session.get_inputs():
                col = meta.name
                if col in row.columns:
                    inp[col] = (
                        row[[col]].astype(np.float32, errors="ignore").to_numpy()
                    )
                else:
                    inp[col] = (
                        row.iloc[:, [int(col.replace("float_input_", ""))]]
                        .astype(np.float32, errors="ignore")
                        .to_numpy()
                    )
            return inp

        backend = "onnx"

        def predict_fn(row):
            probs = session.run(None, _onnx_input(row))[-1]
            if probs.ndim == 2 and probs.shape[1] == 2:
                return probs[0, 1]
            return probs[0] if probs.ndim == 1 else probs[0, 0]
    else:
        backend = "joblib"

        def predict_fn(row):
            return pipeline.predict_proba(row)[0, 1]

    log(f"  Backend: {backend}")

    log("Step 3/9: Loading input data...")
    X = pd.read_csv(input_path)
    missing = [f for f in feature_list if f not in X.columns]
    if missing:
        print(f"[ERROR] Missing features: {missing}", flush=True)
        sys.exit(1)
    X = X[feature_list]
    log(f"  Input shape: {X.shape}")

    log("Step 4/9: System info...")
    print(f"  Device: {platform.platform()}", flush=True)
    print(f"  Python: {platform.python_version()}", flush=True)

    log("Step 5/9: Model size...")
    model_size_kb = model_path.stat().st_size / 1024
    print(f"  Joblib model: {model_size_kb:.2f} KB", flush=True)
    if use_onnx:
        print(f"  ONNX model: {onnx_path.stat().st_size / 1024:.2f} KB", flush=True)

    log("Step 6/9: Warmup...")
    warmup_prob = predict_fn(X.iloc[[0]])
    print(f"  Sample probability: {warmup_prob:.4f}", flush=True)

    log(f"Step 7/9: Running {n_runs} latency tests...")

    def wrapped_predict(row):
        return predict_fn(row)

    latencies = run_latency_test(wrapped_predict, X, n_runs, n_warmup)

    log("Step 8/9: Batch prediction + resource monitoring...")
    power = measure_power()
    batch_probs = [predict_fn(X.iloc[[i]]) for i in range(len(X))]
    batch_preds = [int(p >= threshold) for p in batch_probs]

    proc = psutil.Process(os.getpid())
    ram_mb = round(proc.memory_info().rss / (1024**2), 2)
    cpu = psutil.cpu_percent(interval=1)

    log("Step 9/9: Logging to SQLite...")
    for i in range(min(10, len(batch_probs))):
        log_inference(
            prob=float(batch_probs[i]),
            pred=int(batch_preds[i]),
            risk_level=risk_level(batch_probs[i], threshold),
            latency=latencies,
            backend=backend,
        )

    print("", flush=True)
    print("=" * 46, flush=True)
    print(" FINAL RESULTS", flush=True)
    print("=" * 46, flush=True)

    results = {
        "device": platform.platform(),
        "model_size_kb": round(model_size_kb, 2),
        "input_rows": len(X),
        "threshold": threshold,
        "backend": backend,
        "latency_mean_ms": latencies["mean_ms"],
        "latency_median_ms": latencies["median_ms"],
        "latency_std_ms": latencies["std_ms"],
        "latency_p95_ms": latencies["p95_ms"],
        "latency_p99_ms": latencies["p99_ms"],
        "latency_min_ms": latencies["min_ms"],
        "latency_max_ms": latencies["max_ms"],
        "ram_mb": ram_mb,
        "cpu_percent": cpu,
        **{k: v for k, v in power.items() if v is not None},
        "positive_rate": float(np.mean(batch_preds)),
    }

    for k, v in results.items():
        print(f"  {k}: {v}", flush=True)

    pd.DataFrame([results]).to_csv(results_path, index=False)
    log(f"Results saved to {results_path}")
    log("Edge inference test complete.")


if __name__ == "__main__":
    main()
