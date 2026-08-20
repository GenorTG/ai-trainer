#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "No venv found. Running install..."
    bash install.sh
fi

source .venv/bin/activate
echo "Starting Finetune Studio on http://localhost:7860"
exec python -m finetune_studio "$@"
