@echo off
chcp 65001 > nul
title AI 초등 수행평가 자동 채점 & 평어 생성기
cd /d "%~dp0"

echo ========================================================
echo  🌱 초등 AI 수행평가 자동 채점 및 평어 생성기 실행 중...
echo ========================================================
echo.
echo  * 서버 기동 후 기본 웹 브라우저(http://localhost:8501)가 자동으로 열립니다.
echo  * 프로그램을 종료하시려면 본 창을 닫아주세요.
echo.

if exist "%~dp0python_env\python.exe" (
    "%~dp0python_env\python.exe" launcher.py
    goto end
)

echo [오류] 내장 파이썬 환경(python_env)을 찾을 수 없습니다.
pause

:end
