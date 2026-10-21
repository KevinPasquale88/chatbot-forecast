#!/usr/bin/env bash
# Sets up the environment and launches the clinical risk chatbot.
# Works on macOS and Linux.
#
# Usage:
#   ./run_chatbot.sh                                   # interactive prompts
#   ./run_chatbot.sh --dataset heart_disease --chatbot streamlit
#   ./run_chatbot.sh --dataset diabetes130 --chatbot telegram
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATASET=""
CHATBOT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --chatbot) CHATBOT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "[INFO] Setting up Python virtual environment (env/)..."
if [[ ! -d env ]]; then
    python3 -m venv env
fi
# shellcheck disable=SC1091
source env/bin/activate

echo "[INFO] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [[ -z "$DATASET" ]]; then
    echo ""
    echo "Quale dataset vuoi usare?"
    select choice in "heart_disease" "diabetes130"; do
        if [[ -n "$choice" ]]; then
            DATASET="$choice"
            break
        fi
    done
fi

if [[ -z "$CHATBOT" ]]; then
    echo ""
    echo "Quale tecnologia di chatbot vuoi avviare?"
    select choice in "streamlit" "telegram"; do
        if [[ -n "$choice" ]]; then
            CHATBOT="$choice"
            break
        fi
    done
fi

if [[ "$CHATBOT" == "telegram" && -z "${TELEGRAM_BOT_TOKEN:-}" && ! -f .env ]]; then
    echo ""
    echo "[WARN] Nessun TELEGRAM_BOT_TOKEN trovato in .env o nell'ambiente."
    echo "       Crea un file .env con TELEGRAM_BOT_TOKEN=<il tuo token> prima di continuare."
fi

echo ""
echo "[INFO] Avvio chatbot: dataset=$DATASET, tecnologia=$CHATBOT"
python main.py --dataset "$DATASET" --chatbot "$CHATBOT"
