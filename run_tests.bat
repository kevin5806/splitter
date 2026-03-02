@echo off
setlocal

echo ========================================
echo Running Splitter Unit Tests
echo ========================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    pause
    exit /b 1
)

python -c "import pytest, pytest_cov" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing test dependencies...
    python -m pip install --upgrade pip --quiet
    python -m pip install --quiet -r requirements-dev.txt
    if %errorlevel% neq 0 (
        echo ERROR: Failed to install test dependencies.
        pause
        exit /b 1
    )
)

python -m pytest -v --cov=src.splitter_with_per_image --cov=src.splitter_core --cov=src.splitter_models --cov-report=term-missing tests
if %errorlevel% neq 0 (
    echo.
    echo Some tests failed.
    pause
    exit /b 1
)

echo.
echo All tests passed.
pause
