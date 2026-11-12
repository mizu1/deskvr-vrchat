@echo off
cd /d "%~dp0"
python run.py %*
echo.
echo vdvr exited. Press any key to close this window.
pause >nul
