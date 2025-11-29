@echo off
echo =========================================
echo   HE THONG TIM KIEM XE MAY CU
echo   Starting Streamlit App...
echo =========================================
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found
    echo Using global Python installation
)

echo.
echo Starting app on http://localhost:8503
echo Press Ctrl+C to stop the server
echo.

streamlit run final_app.py --server.port 8503

pause
