@echo off
REM Runs the two offline experiments. Judge reliability is interactive, so it
REM is not included here - run it directly when you are ready to score by hand.
setlocal
set PY=%~dp0..\venv\Scripts\python.exe
set PYTHONIOENCODING=utf-8
"%PY%" "%~dp0experiments\threshold_sensitivity.py"
echo.
"%PY%" "%~dp0experiments\dedup_sensitivity.py"
pause
