@echo off
REM ============================================================
REM  UFC-ML-Predictor - run a prediction
REM ============================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" ".\src\current_model\current_prod_model.py"

echo.
pause
