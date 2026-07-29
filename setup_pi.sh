#!/usr/bin/env bash
set -euo pipefail

echo "=============================================="
echo " Edge AI Raspberry Pi Setup"
echo "=============================================="

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$PROJECT_DIR"

# System packages
echo "[1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip \
    python3-venv \
    sqlite3 \
    || true

# Python virtual environment
echo "[2/5] Creating Python virtual environment..."
python3 -m venv "$PROJECT_DIR/venv"
source "$PROJECT_DIR/venv/bin/activate"

# Python packages
echo "[3/5] Installing Python packages..."
pip install --upgrade pip -q
pip install pandas numpy scikit-learn joblib psutil -q
pip install onnxruntime gradio plotly -q

# Install the edge_ai package (needed by run_edge_inference.py)
pip install -e "$PROJECT_DIR" -q

# Optional: SHAP (larger install, may be slow on Pi)
if pip install shap -q 2>/dev/null; then
    echo "  SHAP installed successfully."
else
    echo "  SHAP install skipped (will use coefficient fallback)."
fi

# Database
echo "[4/5] Initializing monitoring database..."
python3 -c "
from edge_ai.monitoring.metrics import setup_database
print(f'Database ready at: {setup_database()}')
" 2>/dev/null || python3 -c "
import sqlite3, os
db = os.path.expanduser('~/.edge_ai_monitoring.db')
conn = sqlite3.connect(db)
conn.execute('''CREATE TABLE IF NOT EXISTS inference_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    probability REAL, prediction INTEGER, risk_level TEXT,
    latency_mean_ms REAL, latency_p95_ms REAL, latency_p99_ms REAL,
    latency_std_ms REAL, ram_mb REAL, cpu_percent REAL,
    core_volts REAL, power_watts REAL, throttled TEXT, temp_celsius REAL,
    model_backend TEXT, extra TEXT)''')
conn.execute('''CREATE TABLE IF NOT EXISTS system_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    ram_total_mb REAL, ram_available_mb REAL, ram_used_mb REAL,
    cpu_percent REAL, cpu_count INTEGER,
    disk_used_gb REAL, disk_free_gb REAL,
    temperature_celsius REAL, core_voltage REAL, power_watts REAL, throttled TEXT)''')
conn.commit(); conn.close()
print(f'Database ready at: {db}')
"

# Model files
echo "[5/5] Checking model files..."
if [ -d "$MODEL_DIR" ] && ls "$MODEL_DIR"/*.joblib 1>/dev/null 2>&1; then
    echo "  Model files found in $MODEL_DIR"
else
    echo "  WARNING: No model files found in $MODEL_DIR"
    echo "  Copy your .joblib, .onnx, and .csv model artifacts to: $MODEL_DIR"
fi

echo ""
echo "=============================================="
echo " Setup complete!"
echo ""
echo " To run the inference test:"
echo "   source venv/bin/activate"
echo "   python run_edge_inference.py"
echo ""
echo " To launch the dashboard:"
echo "   source venv/bin/activate"
echo "   edge-dashboard"
echo "=============================================="
