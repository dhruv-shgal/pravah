@echo off
echo ========================================
echo   ECO-CONNECT FULL STACK LAUNCHER
echo ========================================
echo.

cd /d "%~dp0"

echo [1/6] Checking Python installation...
python --version
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)
echo ✅ Python is installed

echo.
echo [2/6] Installing/Checking required packages...

echo   - Checking FastAPI...
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo   Installing FastAPI...
    pip install fastapi uvicorn python-multipart
)

echo   - Checking Computer Vision packages...
python -c "import cv2" 2>nul
if errorlevel 1 (
    echo   Installing OpenCV...
    pip install opencv-python
)

python -c "import numpy" 2>nul
if errorlevel 1 (
    echo   Installing NumPy...
    pip install numpy
)

python -c "from PIL import Image" 2>nul
if errorlevel 1 (
    echo   Installing Pillow...
    pip install pillow
)

echo   - Checking AI/ML packages...
python -c "from ultralytics import YOLO" 2>nul
if errorlevel 1 (
    echo   Installing YOLO (Ultralytics)...
    pip install ultralytics
)

python -c "import torch" 2>nul
if errorlevel 1 (
    echo   Installing PyTorch...
    pip install torch torchvision
)

python -c "import insightface" 2>nul
if errorlevel 1 (
    echo   Installing InsightFace...
    pip install insightface onnxruntime
)

echo   - Checking Authentication packages...
python -c "import bcrypt" 2>nul
if errorlevel 1 (
    echo   Installing bcrypt...
    pip install bcrypt
)

python -c "from jose import jwt" 2>nul
if errorlevel 1 (
    echo   Installing python-jose...
    pip install python-jose[cryptography]
)

python -c "from passlib.context import CryptContext" 2>nul
if errorlevel 1 (
    echo   Installing passlib...
    pip install passlib[bcrypt]
)

echo ✅ All packages ready

echo.
echo [3/6] Checking YOLO model files...
if exist "best_yolov11_plantation.pt" (
    echo ✅ Plantation model found
) else (
    echo ⚠️  Plantation model not found: best_yolov11_plantation.pt
)

if exist "best_yolov11_waste_management.pt" (
    echo ✅ Waste management model found
) else (
    echo ⚠️  Waste management model not found: best_yolov11_waste_management.pt
)

if exist "animal_feeding_yolov11.pt" (
    echo ✅ Animal feeding model found
) else (
    echo ⚠️  Animal feeding model not found: animal_feeding_yolov11.pt
)

echo.
echo [4/6] Checking HTML files...
if exist "eco-connect-site.html" (
    echo ✅ Main site found
) else (
    echo ⚠️  Main site not found: eco-connect-site.html
)

if exist "login.html" (
    echo ✅ Login page found
) else (
    echo ⚠️  Login page not found: login.html
)

if exist "signup.html" (
    echo ✅ Signup page found
) else (
    echo ⚠️  Signup page not found: signup.html
)

echo.
echo [5/6] Starting Eco-Connect Server...
echo ========================================
echo 🌐 Frontend: http://localhost:8000
echo 📡 Backend API: http://localhost:8000/api/*
echo 📚 API Docs: http://localhost:8000/docs
echo 📊 Health Check: http://localhost:8000/health
echo.
echo 🎯 Available APIs:
echo   • /api/verify/plantation - Plantation verification
echo   • /api/verify/waste - Waste collection verification
echo   • /api/verify/animal - Animal feeding verification
echo   • /api/verify_task - General verification (routes to above)
echo   • /api/verify-task - Alias for frontend compatibility
echo ========================================
echo.
echo [6/6] Opening browser in 3 seconds...
echo Press Ctrl+C to stop the server
echo.

REM Wait 3 seconds then open browser
timeout /t 3 /nobreak >nul
start http://localhost:8000/eco-connect-site.html

REM Start the FastAPI server
echo 🚀 Starting FastAPI server...
python fastapi2.py

echo.
echo Server stopped. Press any key to exit...
pause >nul
