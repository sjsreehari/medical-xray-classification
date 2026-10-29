@echo off
echo =========================================================
echo   Starting PneumoAI Pediatric Chest X-Ray AI System       
echo =========================================================

echo [1/2] Starting FastAPI Backend on http://localhost:8000 ...
start "FastAPI Backend" python -m uvicorn api:app --host 0.0.0.0 --port 8000

echo Waiting 3 seconds for FastAPI to initialize...
timeout /t 3 /nobreak > nul

echo [2/2] Starting React Frontend on http://localhost:3000 ...
cd frontend
npm run dev

pause
