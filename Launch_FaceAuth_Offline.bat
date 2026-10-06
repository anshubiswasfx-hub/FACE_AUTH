@echo off
title FaceAuth AI - Offline Local Launcher
color 0b
echo ========================================================
echo          FaceAuth AI - Enterprise Biometrics
echo ========================================================
echo.
echo [*] Starting local offline server...
echo [*] Python: C:\Users\borse\AppData\Local\PyEmbed311\python.exe
echo [*] App URL: http://localhost:8501
echo.

set PYTHONIOENCODING=utf-8
set PATH=C:\Users\borse\AppData\Local\PyEmbed311\Scripts;%PATH%

start "" http://localhost:8501
"C:\Users\borse\AppData\Local\PyEmbed311\Scripts\streamlit.exe" run app.py --server.headless false --server.port 8501

pause
