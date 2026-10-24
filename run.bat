@echo off
rem ===================================================================
rem  Engine Simulator - launcher
rem  Запускает симулятор двигателя из этой папки.
rem  Можно передать обычные ключи, например:
rem      run.bat --preset v8_crossplane
rem      run.bat --no-audio
rem      run.bat --audio-backend winmm
rem      run.bat --selftest
rem ===================================================================
setlocal enableextensions
chcp 65001 >nul
title Engine Simulator

rem --- всегда работаем из папки, где лежит этот файл -------------------
cd /d "%~dp0"

if not exist "simulator\__main__.py" (
    echo.
    echo [!] Рядом с этим файлом нет папки "simulator".
    echo     Файл запуска должен лежать в корне проекта.
    echo.
    pause
    exit /b 1
)

rem --- ищем Python, в котором есть numpy, Pillow и tkinter ------------
rem  (простой "py" может указывать на свежий Python без библиотек,
rem   поэтому каждый кандидат проверяется импортом)
set "PY="
call :probe python.exe
call :probe "%ProgramFiles%\Python312\python.exe"
call :probe "%ProgramFiles%\Python313\python.exe"
call :probe "%ProgramFiles%\Python314\python.exe"
call :probe "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY call :probe "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PY call :probe "%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if not defined PY call :probe py.exe

if not defined PY goto :nopython

echo Запускаю симулятор...  ^(Python: %PY%^)
echo.
%PY% -m simulator %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo [!] Симулятор завершился с кодом %RC%.
    pause
)
exit /b %RC%

rem -------------------------------------------------------------------
:nopython
echo.
echo [!] Не найден Python с библиотеками numpy, Pillow и tkinter.
echo.
echo     Поставьте Python 3.10+ и выполните один раз:
echo         python -m pip install -r requirements.txt
echo.
echo     Если Python установлен в необычное место, допишите путь
echo     к python.exe в список "call :probe" внутри этого файла.
echo.
pause
exit /b 1

:probe
if defined PY goto :eof
"%~1" -c "import numpy, PIL, tkinter" >nul 2>&1
rem  кавычки сохраняем: путь может содержать пробелы ("Program Files")
if not errorlevel 1 set PY="%~1"
goto :eof
