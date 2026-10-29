#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

cleanup() {
    echo ""
    echo "Shutting down FastAPI and React frontend..."
    if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
        kill "$API_PID" 2>/dev/null || true
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

echo "========================================================="
echo "  Starting PneumoAI Pediatric Chest X-Ray AI System      "
echo "========================================================="

# 1. Start FastAPI backend
echo "[1/2] Starting FastAPI backend on http://localhost:8000 ..."
python -m uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

sleep 3

# 2. Start React frontend
echo "[2/2] Starting React frontend on http://localhost:3000 ..."
cd frontend
npm run dev
