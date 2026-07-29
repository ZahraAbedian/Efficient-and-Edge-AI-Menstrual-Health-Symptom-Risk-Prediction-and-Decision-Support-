import os
import json
import tempfile
from pathlib import Path

import pytest

from edge_ai.monitoring.metrics import (
    setup_database, measure_resources, log_inference,
    get_recent_inferences, log_system_snapshot, get_recent_snapshots,
)


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    setup_database(path)
    yield path
    os.unlink(path)


def test_setup_database(db_path):
    assert Path(db_path).exists()


def test_log_and_retrieve_inference(db_path):
    latency = {"mean_ms": 5.0, "p95_ms": 7.0, "p99_ms": 9.0, "std_ms": 1.0}
    row_id = log_inference(
        prob=0.75, pred=1, risk_level="High",
        latency=latency, backend="joblib",
        extra={"test": True}, db_path=db_path,
    )
    assert row_id > 0

    df = get_recent_inferences(limit=10, db_path=db_path)
    assert len(df) == 1
    assert df.iloc[0]["probability"] == 0.75
    assert df.iloc[0]["risk_level"] == "High"


def test_log_and_retrieve_snapshot(db_path):
    row_id = log_system_snapshot(db_path=db_path)
    assert row_id > 0

    df = get_recent_snapshots(limit=10, db_path=db_path)
    assert len(df) == 1
    assert "ram_total_mb" in df.columns


def test_measure_resources():
    resources = measure_resources()
    assert "ram_total_mb" in resources
    assert resources["ram_total_mb"] > 0
    assert resources["cpu_count"] > 0
