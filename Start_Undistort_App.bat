@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m undistort_app %*
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m undistort_app %*
) else (
    python -m undistort_app %*
)
