@echo off
chcp 65001 > nul
echo ========================================================
echo  초등 AI 수행평가 자동 채점 및 평어 생성기 실행 중...
echo ========================================================
echo.
C:\Python311\python.exe -m streamlit run app.py
pause
