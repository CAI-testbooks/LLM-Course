@echo off
setlocal
cd /d "%~dp0"

:: Check for venv and activate if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found, running with system python...
)

echo Starting Web UI...
start "" python web_ui.py
timeout /t 3 >nul
echo Opening Browser...
start "" http://127.0.0.1:5000/
