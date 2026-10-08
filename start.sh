#!/bin/bash
set -e

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found in $SCRIPT_DIR/.venv"
    echo "Creating virtual environment and installing dependencies..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

PYTHON=".venv/bin/python"
STREAMLIT=".venv/bin/streamlit"

echo "=================================================="
echo "  Starting Chest X-Ray Disease Detection System   "
echo "=================================================="

# Function to clean up background processes on exit (Ctrl+C)
cleanup() {
    echo ""
    echo "Shutting down FastAPI and Streamlit..."
    if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
        kill "$API_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Start FastAPI backend server in background
echo "[1/2] Starting FastAPI server on http://localhost:8000 ..."
$PYTHON api.py &
API_PID=$!

# Wait for API server to become ready
echo "Waiting for API server to be healthy..."
for i in {1..30}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "✅ FastAPI server is online (PID: $API_PID)"
        break
    fi
    sleep 1
done

# 2. Start Streamlit frontend
echo "[2/2] Starting Streamlit app on http://localhost:8501 ..."
$STREAMLIT run app.py --server.port 8501 --server.headless false

