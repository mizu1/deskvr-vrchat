@echo off
cd /d "%~dp0"
python tools\glassvr_mode.py normal
echo.
echo Close vdvr (python run.py), then restart SteamVR and launch VRChat normally.
pause >nul
