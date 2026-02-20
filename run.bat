@echo off
setlocal enabledelayedexpansion
title SDXL GGUF Quantize Tool

cd /D "%~dp0"

echo ========================================
echo   SDXL GGUF Quantize Tool
echo ========================================

:: Check for python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python (3.10+ recommended) and ensure it is added to your PATH during installation.
    pause
    exit /b
)

:: Check for venv
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Creating one in the "venv" folder...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
    echo [INFO] Virtual environment created successfully.
)

:: Activate venv
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: Install Requirements
echo [INFO] Checking and installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies. Please check your internet connection or requirements.txt.
    pause
    exit /b
)

:: Run the app
echo [INFO] Starting GUI...
python gui.py

echo.
echo Application closed.
pause
