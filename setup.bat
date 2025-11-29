@echo off
echo =========================================
echo   SETUP HE THONG XE MAY CU
echo =========================================
echo.

echo [1/4] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.9-3.11
    pause
    exit /b 1
)
echo.

echo [2/4] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists
) else (
    python -m venv venv
    echo Virtual environment created successfully
)
echo.

echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo.

echo [4/4] Installing dependencies...
pip install -r requirements.txt
echo.

echo =========================================
echo   SETUP COMPLETED SUCCESSFULLY!
echo =========================================
echo.
echo To start the app, run: start_app.bat
echo Or manually: streamlit run final_app.py
echo.
pause
