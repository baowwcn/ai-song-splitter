@echo off
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
title Song Splitter Service
cd /d "%~dp0"

rem locate ffmpeg under tools\ffmpeg\<version>\bin
set "FFDIR="
for /d %%d in ("tools\ffmpeg\*") do if exist "%%d\bin\ffmpeg.exe" set "FFDIR=%%d\bin"
if defined FFDIR set "PATH=%PATH%;%FFDIR%"

rem portable runtime first, fall back to python on PATH (venv activated or system install)
set "PY="
if exist "runtime\python\python.exe" set "PY=runtime\python\python.exe"
if not defined PY (
  for %%p in (".venv\Scripts\python.exe" "venv\Scripts\python.exe") do if not defined PY if exist %%p set "PY=%%~p"
)
if not defined PY for /f "delims=" %%p in ('where python 2^>nul') do if not defined PY set "PY=%%p"

if not defined PY (
  echo [ERROR] 找不到 Python。
  echo   便携版:runtime\python\python.exe
  echo   或先建虚拟环境:.venv\Scripts\python.exe
  echo   或把 python 装好并加入 PATH。
  pause
  exit /b 1
)

echo ============================================
echo   Song Splitter Service
echo   Python: %PY%
echo   URL: http://127.0.0.1:8000
echo   Close this window to stop the service
echo ============================================
echo.

rem open browser automatically after 3 seconds
start "" /b cmd /c "timeout /t 3 /nobreak >nul & start http://127.0.0.1:8000"

"%PY%" app.py

pause
