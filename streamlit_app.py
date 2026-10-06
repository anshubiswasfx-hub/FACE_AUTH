"""
Streamlit Cloud Entry Point wrapper
Redirects execution to app.py
"""
import runpy
import sys
from pathlib import Path

app_path = Path(__file__).parent / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
