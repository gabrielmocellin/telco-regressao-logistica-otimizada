@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

set "VENV_DIR=.venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/3] Criando ambiente virtual em %VENV_DIR%...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERRO] Falha ao criar o ambiente virtual. Verifique se o Python esta instalado e no PATH.
        pause
        exit /b 1
    )
) else (
    echo [1/3] Ambiente virtual encontrado em %VENV_DIR%.
)

echo [2/3] Instalando/atualizando dependencias do requirements.txt...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip --quiet
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERRO] Falha ao instalar as dependencias.
    pause
    exit /b 1
)

echo [3/3] Executando main.py...
echo.
"%VENV_DIR%\Scripts\python.exe" main.py

echo.
pause
