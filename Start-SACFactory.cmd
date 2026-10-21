@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-SACFactory.ps1" -Restart
if errorlevel 1 exit /b %errorlevel%
start "SACFactory" "http://127.0.0.1:8517"
