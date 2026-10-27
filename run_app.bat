@echo off
rem Double-click to start the DR screening interface (see run_app.ps1 for options).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_app.ps1" %*
pause
