#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "[INFO] Starting Conductor environment setup..."

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "[ERROR] '$1' could not be found. Please ensure it is installed and added to your PATH, then try again."
        exit 1
    fi
}

# Check prerequisites
echo "[INFO] Checking prerequisites..."
check_command python
check_command pip
check_command nvcc

echo "[INFO] All prerequisites (python, pip, nvcc) found."

# 1. Install pip dependencies
if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "[WARNING] requirements.txt not found. Skipping."
fi

# 2. Compile CUDA extensions
echo "[INFO] Compiling CUDA extensions in modules/cuda_modules..."
if [ -d "modules/cuda_modules" ]; then
    cd modules/cuda_modules
    
    # Run the setup script to build the C++/CUDA extension in-place
    python setup.py build_ext --inplace
    
    cd ../..
    echo "[INFO] CUDA extensions compiled successfully."
else
    echo "[ERROR] Directory modules/cuda_modules not found. Compilation failed."
    exit 1
fi

echo "[INFO] Setup completed successfully!"
