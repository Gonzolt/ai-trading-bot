@echo off
setlocal
cd /d "%~dp0"

set OMP_NUM_THREADS=20
set MKL_NUM_THREADS=20
set OPENBLAS_NUM_THREADS=20

py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)" >nul 2>&1
if errorlevel 1 (
    echo Python 3.11 is required. Install it from https://www.python.org/downloads/
    exit /b 1
)

if not exist "venv/Scripts/python.exe" (
    echo Creating Python 3.11 virtual environment...
    py -3.11 -m venv venv
    if errorlevel 1 exit /b 1
)

call "venv/Scripts/activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Setup complete. Run run.bat to launch Korvax TradeMind.
endlocal
