@echo off
setlocal
cd /d "%~dp0"
title Voice Input Method

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found:
    echo %CD%\.venv\Scripts\python.exe
    pause
    exit /b 1
)

if not exist "config.windows.yaml" (
    echo [ERROR] Windows configuration not found:
    echo %CD%\config.windows.yaml
    pause
    exit /b 1
)

echo Starting Voice Input Method...
echo Hold CapsLock to record; release it to transcribe.
echo Short CapsLock presses still toggle uppercase.
echo.
".venv\Scripts\python.exe" -u -m voice_input_method --config "%CD%\config.windows.yaml"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%" == "0" (
    echo.
    echo [ERROR] Voice Input Method exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%
