@echo off
REM ============================================================
REM  UFC-ML-Predictor - setup (run once)
REM  Creates a virtual environment and installs all dependencies.
REM ============================================================
cd /d "%~dp0"

echo.
echo [1/4] Checking Python...
python --version
if errorlevel 1 (
    echo.
    echo ERROR: Python not found. Install Python 3.11+ from https://www.python.org/downloads/
    echo Make sure to tick "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

echo.
echo [2/4] Creating virtual environment (.venv)...
python -m venv .venv

echo.
echo [3/4] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo.
echo [4/4] Installing dependencies...
REM requirements.txt covers the scraper; the next line adds the missing ML libs.
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" -m pip install xgboost scikit-learn matplotlib

echo.
echo ============================================================
echo  Setup complete! Double-click run.bat to make a prediction.
echo ============================================================
pause
