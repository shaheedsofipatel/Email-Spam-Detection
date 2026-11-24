@echo off
echo ========================================================
echo   Starting AI Email Spam Detector Web Application...
echo ========================================================
call .venv\Scripts\activate.bat
streamlit run app.py
pause
