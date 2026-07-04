@echo off
setlocal
cd /d "%~dp0"

set OMP_NUM_THREADS=20
set MKL_NUM_THREADS=20
set OPENBLAS_NUM_THREADS=20

if exist "venv/Scripts/python.exe" (
    "venv/Scripts/python.exe" alpaca_trademind.py
) else (
    py -3.11 alpaca_trademind.py
)

endlocal
