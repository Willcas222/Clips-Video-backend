@echo off
cd /d "%~dp0"
.venv\Scripts\python -m uvicorn main:app --port 8000
