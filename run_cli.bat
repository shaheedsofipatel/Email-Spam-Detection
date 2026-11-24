@echo off
echo ========================================================
echo   Starting AI Email Spam Detector CLI Tool...
echo ========================================================
call .venv\Scripts\activate.bat
python predict.py
pause
