@echo off
cd /d "%~dp0"
start "" /min python server.py
%SystemRoot%\System32\timeout.exe /t 3 /nobreak >nul
start "" http://127.0.0.1:8765
exit
