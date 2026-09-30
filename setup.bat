@echo off
REM ===========================================================================
REM  StudyBuddy - Windows setup helper
REM  Installs all required Python packages directly (no requirements.txt),
REM  so a corrupted requirements.txt can never break installation again.
REM ===========================================================================

echo.
echo  Installing StudyBuddy dependencies...
echo.

python -m pip install rich rapidfuzz requests beautifulsoup4 python-dotenv colorama tabulate

if errorlevel 1 (
    echo.
    echo  Install failed. Try:  py -m pip install rich rapidfuzz requests beautifulsoup4 python-dotenv colorama tabulate
    pause
    exit /b 1
)

echo.
echo  Dependencies installed successfully.
echo  Run StudyBuddy with:  python app.py
echo.
pause
