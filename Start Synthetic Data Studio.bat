@echo off
rem Opens the Start / Stop control panel for the dashboard (no console window stays open).
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0launcher.py"
