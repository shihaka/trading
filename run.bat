@echo off
chcp 65001 >nul
setlocal enableextensions enabledelayedexpansion

REM Переходим в папку со скриптами
cd /d "%~dp0"

echo === Автозапуск каждые 5 минут (collector -> signals) ===

:loop
echo.
echo [%date% %time%] START CYCLE

REM 1) Сбор данных (монеты читаются из coins.txt)
python kline_collector.py
if errorlevel 1 (
  echo [%date% %time%] collector завершился с кодом %errorlevel%
)

REM 2) Генерация сигналов (монеты читаются из coins.txt)
python signals.py
if errorlevel 1 (
  echo [%date% %time%] signals завершился с кодом %errorlevel%
)

echo [%date% %time%] CYCLE DONE. Ждем 5 минут...
timeout /t 300 /nobreak >nul
goto loop
