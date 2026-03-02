@echo off
setlocal

echo ========================================
echo Build Splitter EXE
echo ========================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Install Python 3.x from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Installing build dependencies...
python -m pip install --upgrade pip --quiet
python -m pip install --quiet -r requirements.txt pyinstaller pyinstaller-hooks-contrib
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo Building executable with PyInstaller...
python -m PyInstaller ^
    --noconfirm ^
    --clean ^
    --onefile ^
    --windowed ^
    --name Splitter ^
    --collect-all tkinterdnd2 ^
    --collect-all sv_ttk ^
    --collect-all cv2 ^
    splitter_with_per_image.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed.
    pause
    exit /b 1
)

echo.
echo Build complete.
echo EXE path: dist\Splitter.exe
pause
