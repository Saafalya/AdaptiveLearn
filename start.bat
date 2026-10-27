@echo off
title AdaptiveLearn
cd /d "%~dp0"
echo === AdaptiveLearn - AI-Based Personalized Learning Recommendation System ===
echo.

rem Find Python: the "py" launcher first, then "python"
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (where python >nul 2>&1 && set "PY=python")
if not defined PY (
    echo [X] Python is missing. Install Python 3.10 or newer from https://python.org
    echo     During setup, tick "Add python.exe to PATH", then run this file again.
    pause
    exit /b 1
)

rem Keep packages in a private folder so they don't clash with other projects
if not exist ".venv\Scripts\python.exe" (
    echo Creating a private Python environment in .venv ...
    %PY% -m venv .venv || (echo [X] Could not create .venv. & pause & exit /b 1)
)
call ".venv\Scripts\activate.bat"
python -m pip install -q --upgrade pip >nul 2>&1

rem Anything typed after start.bat is passed on, e.g.  start.bat --quick
python run_website.py %*

echo.
pause
