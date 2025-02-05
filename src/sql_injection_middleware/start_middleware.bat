@echo off
cd /d %~dp0
cd ..
D:\Anaconda\python.exe -m sql_injection_middleware.start_middleware
pause
