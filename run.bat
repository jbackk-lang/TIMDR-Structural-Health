@echo off
setlocal
title TIMDR Structural Health - Dashboard
echo ============================================================
echo  TIMDR Structural Health - uruchamianie dashboardu
echo ============================================================
where python >nul 2>nul
if %ERRORLEVEL%==0 (set PYCMD=python) else (
  where py >nul 2>nul
  if %ERRORLEVEL%==0 (set PYCMD=py) else (
    echo [BLAD] Nie znaleziono Pythona. Pobierz z https://www.python.org/downloads/ i zaznacz "Add python.exe to PATH".
    pause & exit /b 1))
echo Uzywam: %PYCMD%
echo Instaluje zaleznosci (numpy, scipy)...
%PYCMD% -m pip install --quiet numpy scipy
if %ERRORLEVEL% NEQ 0 (echo [BLAD] Instalacja zaleznosci nie powiodla sie. & pause & exit /b 1)
echo Dashboard: http://127.0.0.1:8765  (zamknij to okno, zeby zatrzymac)
start "" http://127.0.0.1:8765
%PYCMD% dashboard.py
if %ERRORLEVEL% NEQ 0 (echo. & echo [BLAD] Serwer zakonczyl sie bledem - tresc powyzej. & pause)
