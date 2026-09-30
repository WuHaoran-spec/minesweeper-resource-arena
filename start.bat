@echo off
cd /d "%~dp0"
python -c "import numpy" >nul 2>nul
if errorlevel 1 python -m pip install -r requirements.txt
start "" http://127.0.0.1:8765
python -m arena.server
pause
