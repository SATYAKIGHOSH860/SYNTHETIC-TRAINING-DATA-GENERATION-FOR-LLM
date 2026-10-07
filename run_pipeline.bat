@echo off
REM Runs the full pipeline. Edit --max-chunks to change the sample size.
REM The venv lives one level up, shared with the other project folders.
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
"%PY%" "%~dp0main.py" --max-chunks 100 %*
pause
