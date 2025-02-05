@echo off
cd /d %~dp0
cd ..\..
D:\Anaconda\python.exe -m uvicorn demo_backend_service.main:app --reload --app-dir src --host 0.0.0.0 --port 8000
pause
