@echo off

rem =====================================================
rem 请根据你的Redis安装路径修改以下命令中的路径
rem 默认路径：E:\Program Files\redis\redis-windows-7.2.5
rem 修改这里指向你的Redis安装目录的start.bat位置
rem =====================================================

echo Starting Redis Server...
start "Redis Server" cmd /c "cd /d "E:\Program Files\redis\redis-windows-7.2.5" && start.bat"

echo Waiting for Redis to start...
timeout /t 3

echo Starting SQL Injection Middleware...
cd /d %~dp0
cd ..
D:\Anaconda\python.exe -m sql_injection_middleware.start_middleware
pause
