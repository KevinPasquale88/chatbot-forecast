@echo off
REM Sets up the environment and launches the clinical risk chatbot on Windows (cmd.exe).
REM
REM Usage:
REM   run_chatbot.bat
REM   run_chatbot.bat --dataset heart_disease --chatbot streamlit
REM   run_chatbot.bat --dataset diabetes130 --chatbot telegram
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "DATASET="
set "CHATBOT="

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--dataset" (
    set "DATASET=%~2"
    shift
    shift
    goto parse_args
)
if /i "%~1"=="--chatbot" (
    set "CHATBOT=%~2"
    shift
    shift
    goto parse_args
)
echo Unknown argument: %~1
exit /b 1
:args_done

echo [INFO] Setting up Python virtual environment (env\)...
if not exist env (
    python -m venv env
)
call env\Scripts\activate.bat

echo [INFO] Installing dependencies...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

if "%DATASET%"=="" (
    echo.
    echo Quale dataset vuoi usare?
    echo   1^) heart_disease
    echo   2^) diabetes130
    set /p DS_CHOICE="Scelta [1-2]: "
    if "!DS_CHOICE!"=="1" set "DATASET=heart_disease"
    if "!DS_CHOICE!"=="2" set "DATASET=diabetes130"
)

if "%CHATBOT%"=="" (
    echo.
    echo Quale tecnologia di chatbot vuoi avviare?
    echo   1^) streamlit
    echo   2^) telegram
    set /p CB_CHOICE="Scelta [1-2]: "
    if "!CB_CHOICE!"=="1" set "CHATBOT=streamlit"
    if "!CB_CHOICE!"=="2" set "CHATBOT=telegram"
)

if /i "%CHATBOT%"=="telegram" if not exist .env (
    echo.
    echo [WARN] Nessun file .env trovato.
    echo        Crea un file .env con TELEGRAM_BOT_TOKEN=il_tuo_token prima di continuare.
)

echo.
echo [INFO] Avvio chatbot: dataset=%DATASET%, tecnologia=%CHATBOT%
python main.py --dataset %DATASET% --chatbot %CHATBOT%
