import os
import time
import platform
import sqlite3
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional

import psutil
import numpy as np
import pandas as pd


DB_PATH = Path.home() / ".edge_ai_monitoring.db"


def setup_database(db_path: str | Path = DB_PATH) -> str:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inference_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            probability REAL,
            prediction INTEGER,
            risk_level TEXT,
            latency_mean_ms REAL,
            latency_p95_ms REAL,
            latency_p99_ms REAL,
            latency_std_ms REAL,
            ram_mb REAL,
            cpu_percent REAL,
            core_volts REAL,
            power_watts REAL,
            throttled TEXT,
            temp_celsius REAL,
            model_backend TEXT,
            extra TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS system_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ram_total_mb REAL,
            ram_available_mb REAL,
            ram_used_mb REAL,
            cpu_percent REAL,
            cpu_count INTEGER,
            disk_used_gb REAL,
            disk_free_gb REAL,
            temperature_celsius REAL,
            core_voltage REAL,
            power_watts REAL,
            throttled TEXT
        )
    """)
    for table, col in [("inference_logs", "power_watts"),
                       ("system_snapshots", "power_watts")]:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} REAL")
        except Exception:
            pass
    conn.commit()
    conn.close()
    return str(db_path)


def measure_latency(
    predict_fn,
    sample,
    n_runs: int = 100,
    warmup: int = 5,
) -> dict[str, float]:
    for _ in range(warmup):
        predict_fn(sample)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(sample)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    arr = np.array(latencies)
    return {
        "mean_ms": float(round(arr.mean(), 4)),
        "median_ms": float(round(np.median(arr), 4)),
        "std_ms": float(round(arr.std(), 4)),
        "p95_ms": float(round(np.percentile(arr, 95), 4)),
        "p99_ms": float(round(np.percentile(arr, 99), 4)),
        "min_ms": float(round(arr.min(), 4)),
        "max_ms": float(round(arr.max(), 4)),
    }


_RAPL_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"
_prev_rapl: Optional[tuple[float, float]] = None


def _read_rapl() -> Optional[float]:
    try:
        with open(_RAPL_PATH) as f:
            return int(f.read().strip()) * 1e-6
    except Exception:
        return None


def _read_rapl_power() -> Optional[float]:
    global _prev_rapl
    now = time.time()
    energy = _read_rapl()
    if energy is None:
        return None
    if _prev_rapl is not None:
        t_prev, e_prev = _prev_rapl
        dt = now - t_prev
        if dt > 0.01:
            power = (energy - e_prev) / dt
            _prev_rapl = (now, energy)
            return round(power, 3)
    _prev_rapl = (now, energy)
    return None


def _read_thermal() -> Optional[float]:
    for path in [
        "/sys/class/thermal/thermal_zone0/temp",
        "/sys/class/hwmon/hwmon0/temp1_input",
    ]:
        try:
            with open(path) as f:
                val = int(f.read().strip())
                return val / 1000.0
        except Exception:
            continue
    return None


def measure_power() -> dict[str, Optional[float | str]]:
    result = {
        "core_volts": None,
        "power_watts": None,
        "throttled": None,
        "temp_celsius": None,
    }

    for cmd, key in [
        ("vcgencmd measure_volts core", "core_volts"),
        ("vcgencmd get_throttled", "throttled"),
        ("vcgencmd measure_temp", "temp_celsius"),
    ]:
        try:
            out = os.popen(cmd).read().strip()
            if "volt" in out:
                result[key] = float(out.replace("volt=", "").replace("V", ""))
            elif "throttled" in out:
                result[key] = out.replace("throttled=", "")
            elif "temp" in out:
                result[key] = float(out.replace("temp=", "").replace("'C", ""))
        except Exception:
            pass

    if result["temp_celsius"] is None:
        result["temp_celsius"] = _read_thermal()

    rapl_power = _read_rapl_power()
    if rapl_power is not None:
        result["power_watts"] = rapl_power

    return result


def measure_resources() -> dict[str, float | int]:
    import psutil

    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    cpu_percent = psutil.cpu_percent(interval=0.5)

    return {
        "ram_total_mb": round(mem.total / (1024**2), 2),
        "ram_available_mb": round(mem.available / (1024**2), 2),
        "ram_used_mb": round(mem.used / (1024**2), 2),
        "cpu_percent": cpu_percent,
        "cpu_count": psutil.cpu_count(),
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_free_gb": round(disk.free / (1024**3), 2),
    }


def measure_all() -> dict:
    power = measure_power()
    resources = measure_resources()
    return {**power, **resources}


def log_inference(
    prob: float,
    pred: int,
    risk_level: str,
    latency: dict[str, float],
    backend: str = "joblib",
    extra: Optional[dict] = None,
    db_path: str | Path = DB_PATH,
) -> int:
    power = measure_power()
    proc = psutil.Process(os.getpid())

    power_watts = power.get("power_watts")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        """INSERT INTO inference_logs
        (timestamp, probability, prediction, risk_level,
         latency_mean_ms, latency_p95_ms, latency_p99_ms, latency_std_ms,
         ram_mb, cpu_percent, core_volts, power_watts, throttled, temp_celsius,
         model_backend, extra)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(UTC).isoformat(),
            prob,
            pred,
            risk_level,
            latency.get("mean_ms"),
            latency.get("p95_ms"),
            latency.get("p99_ms"),
            latency.get("std_ms"),
            round(proc.memory_info().rss / (1024**2), 2),
            psutil.cpu_percent(interval=0.1),
            power.get("core_volts"),
            power_watts,
            power.get("throttled"),
            power.get("temp_celsius"),
            backend,
            json.dumps(extra) if extra else None,
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def log_system_snapshot(db_path: str | Path = DB_PATH) -> int:
    power = measure_power()
    resources = measure_resources()

    power_watts = power.get("power_watts")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute(
        """INSERT INTO system_snapshots
        (timestamp, ram_total_mb, ram_available_mb, ram_used_mb,
         cpu_percent, cpu_count, disk_used_gb, disk_free_gb,
         temperature_celsius, core_voltage, power_watts, throttled)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(UTC).isoformat(),
            resources["ram_total_mb"],
            resources["ram_available_mb"],
            resources["ram_used_mb"],
            resources["cpu_percent"],
            resources["cpu_count"],
            resources["disk_used_gb"],
            resources["disk_free_gb"],
            power.get("temp_celsius"),
            power.get("core_volts"),
            power_watts,
            power.get("throttled"),
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_recent_inferences(
    limit: int = 100,
    db_path: str | Path = DB_PATH,
) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        f"SELECT * FROM inference_logs ORDER BY timestamp DESC LIMIT {limit}",
        conn,
    )
    conn.close()
    return df


def get_recent_snapshots(
    limit: int = 100,
    db_path: str | Path = DB_PATH,
) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        f"SELECT * FROM system_snapshots ORDER BY timestamp DESC LIMIT {limit}",
        conn,
    )
    conn.close()
    return df
