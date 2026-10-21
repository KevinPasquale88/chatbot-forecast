# Sets up the environment and launches the clinical risk chatbot on Windows (PowerShell).
#
# Usage:
#   .\run_chatbot.ps1
#   .\run_chatbot.ps1 -Dataset heart_disease -Chatbot streamlit
#   .\run_chatbot.ps1 -Dataset diabetes130 -Chatbot telegram
param(
    [ValidateSet("heart_disease", "diabetes130")]
    [string]$Dataset,

    [ValidateSet("streamlit", "telegram")]
    [string]$Chatbot
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "[INFO] Setting up Python virtual environment (env\)..."
if (-not (Test-Path "env")) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) { $pythonCmd = Get-Command py -ErrorAction SilentlyContinue }
    if (-not $pythonCmd) { throw "Python non trovato nel PATH." }
    & $pythonCmd.Source -m venv env
}
& .\env\Scripts\Activate.ps1

Write-Host "[INFO] Installing dependencies..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

if (-not $Dataset) {
    Write-Host ""
    Write-Host "Quale dataset vuoi usare?"
    Write-Host "  1) heart_disease"
    Write-Host "  2) diabetes130"
    $choice = Read-Host "Scelta [1-2]"
    $Dataset = if ($choice -eq "2") { "diabetes130" } else { "heart_disease" }
}

if (-not $Chatbot) {
    Write-Host ""
    Write-Host "Quale tecnologia di chatbot vuoi avviare?"
    Write-Host "  1) streamlit"
    Write-Host "  2) telegram"
    $choice = Read-Host "Scelta [1-2]"
    $Chatbot = if ($choice -eq "2") { "telegram" } else { "streamlit" }
}

if ($Chatbot -eq "telegram" -and -not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "[WARN] Nessun file .env trovato."
    Write-Host "       Crea un file .env con TELEGRAM_BOT_TOKEN=il_tuo_token prima di continuare."
}

Write-Host ""
Write-Host "[INFO] Avvio chatbot: dataset=$Dataset, tecnologia=$Chatbot"
python main.py --dataset $Dataset --chatbot $Chatbot
