@echo off
cd /d "%~dp0"
python tools\glassvr_mode.py vdvr
echo.
echo Restart SteamVR, then run start.bat and launch_vrchat_vr.bat.
pause >nul
