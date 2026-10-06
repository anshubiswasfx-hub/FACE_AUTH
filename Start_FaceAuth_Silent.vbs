' FaceAuth AI - Silent Background Launcher (No Terminal Window)
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\borse\.gemini\antigravity\scratch\FACE_AUTH"
WshShell.Run "cmd /c set PYTHONIOENCODING=utf-8 && start http://localhost:8501 && C:\Users\borse\AppData\Local\PyEmbed311\Scripts\streamlit.exe run app.py --server.headless true --server.port 8501", 0, False
