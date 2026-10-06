@echo off
title Stop FaceAuth AI
color 0c
echo Stopping any running FaceAuth AI / Streamlit instances...
taskkill /f /im streamlit.exe >nul 2>&1
echo Done! All FaceAuth AI processes stopped.
timeout /t 2 >nul
