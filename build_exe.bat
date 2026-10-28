@echo off
rem ===================================================================
rem  Build a portable EngineLab folder for Windows x64.
rem
rem    build_exe.bat            normal build
rem    build_exe.bat --zip      also write dist\EngineLab-portable.zip
rem
rem  Requirements on THIS machine: Windows 10/11 x64, Python 3.12-3.14,
rem  and the runtime dependencies (pip install -r requirements.txt).
rem  The target machine needs none of that -- just copy the folder.
rem ===================================================================
setlocal enableextensions
cd /d "%~dp0"

set "PY=python"
%PY% -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [i] PyInstaller is not installed, installing build requirements...
    %PY% -m pip install -r requirements-dev.txt
    if errorlevel 1 goto :failed
)

echo.
echo [1/2] running the test suite...
%PY% -m pytest tests -q
if errorlevel 1 (
    echo.
    echo [!] tests failed - refusing to ship a broken build.
    goto :failed
)

echo.
echo [2/2] freezing...
%PY% -m PyInstaller --noconfirm --clean engine_lab.spec
if errorlevel 1 goto :failed

if /i "%~1"=="--zip" (
    if exist "dist\EngineLab-portable.zip" del /q "dist\EngineLab-portable.zip"
    powershell -NoProfile -Command "Compress-Archive -Path 'dist\EngineLab' -DestinationPath 'dist\EngineLab-portable.zip' -CompressionLevel Optimal"
    echo.
    echo [ok] dist\EngineLab-portable.zip
)

echo.
echo [ok] portable build: dist\EngineLab\EngineLab.exe
echo      copy the whole dist\EngineLab folder to the other PC.
exit /b 0

:failed
echo.
echo [!] build failed.
pause
exit /b 1
