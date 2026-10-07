@echo off
REM Launches the dashboard. This is the main way to use the project:
REM upload PDFs, press START, and watch it run. No command line needed.
setlocal
set PY=%~dp0..\venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
if not exist "%PY%" (
  echo Could not find the virtual environment at:
  echo   %PY%
  echo Create one with:  python -m venv venv ^&^& venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)
echo Starting the dashboard - your browser will open shortly.
"%PY%" -m streamlit run "%~dp0dashboard\app.py"
pause
