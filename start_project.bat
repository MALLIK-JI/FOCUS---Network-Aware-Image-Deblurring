@echo off
title Launching Image Deblurring Suite...
echo ========================================================
echo   Network-Aware Adaptive Image Deblurring Suite
echo   B.Tech Capstone Project
echo ========================================================
echo.

set PROJ_DIR=C:\Users\TANISHK RAJ\OneDrive\Desktop\kanix\Image_Deblurring_Project
set VENV_PY=%PROJ_DIR%\deblur_env\Scripts\python.exe
set STREAMLIT_BIN=%PROJ_DIR%\deblur_env\Scripts\streamlit.exe
set APP_DIR=%PROJ_DIR%\Restormer\Motion_Deblurring

echo [1/3] Checking GPU FastAPI Server on port 8000...
powershell -Command "try { $r = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/info' -TimeoutSec 1; Write-Host 'Server is already running on GPU!' } catch { exit 1 }"
if %ERRORLEVEL% NEQ 0 (
    echo Starting GPU FastAPI Server in background...
    start "Deblurring GPU Server (Port 8000)" cmd /k "cd /d "%APP_DIR%" && "%VENV_PY%" -m uvicorn server:app --host 127.0.0.1 --port 8000"
    timeout /t 3 /nobreak >nul
) else (
    echo Server is already online!
)

echo.
echo [2/3] Launching Interactive Streamlit Web UI (Port 8501)...
cd /d "%APP_DIR%"
start "" "%STREAMLIT_BIN%" run app.py --server.port 8501

echo.
echo [3/3] Application launched! Browser will open at:
echo       http://localhost:8501
echo ========================================================
pause
