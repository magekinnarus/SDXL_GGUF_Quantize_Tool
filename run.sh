#!/bin/bash
# SDXL GGUF Quantize Tool - Linux Launcher

# Change directory to the location of this script
cd "$(dirname "$0")"

echo "========================================"
echo "  SDXL GGUF Quantize Tool"
echo "========================================"

# Check for python3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    echo "Please install Python 3.10+ to continue."
    exit 1
fi

# Check for venv
if [ ! -f "venv/bin/activate" ]; then
    echo "[INFO] Virtual environment not found. Creating one in the 'venv' folder..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment. Ensure you have python3-venv installed."
        exit 1
    fi
    echo "[INFO] Virtual environment created successfully."
fi

# Activate venv
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Install Requirements
echo "[INFO] Checking and installing dependencies..."
python3 -m pip install --upgrade pip >/dev/null 2>&1
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies. Please check your internet connection or requirements.txt."
    exit 1
fi

# Run the app
echo "[INFO] Starting GUI..."
python3 gui.py

echo ""
echo "Application closed."
