@echo off
rem Drag a folder of SEM .tif photos onto this file to measure its nanowires.
rem Results go to results\measure\<folder name>\ in this repository.
rem Any extra options (see README.md) can be added after the folder in a terminal:
rem   measure\measure_wires.bat "D:\SEM\run 7" --all-scales
set PYTHONIOENCODING=utf-8
if "%~1"=="" (
    echo Drag a folder of SEM .tif photos onto measure_wires.bat to measure it.
    pause
    exit /b 1
)
"%~dp0..\venv\Scripts\python.exe" "%~dp0measure_wires.py" %*
pause
