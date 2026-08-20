#!/usr/bin/env bash
# AI Trainer — Deploy to genorbox1 production
# Run from fan-dragon after commits.
#
# What gets synced (from fan-dragon ai-trainer/):
#   - server/        → /home/genorbox1/llama-server/inference_server/
#   - tests/         → /home/genorbox1/llama-server/tests/
#   - trainer/ src   → /home/genorbox1/llama-server/finetune_studio/
#
# What does NOT get synced (production stays stable):
#   - serve.py       (production llama-server wrapper, separate)
#   - chris-ai.service  (systemd unit, separate)
#   - GGUF models    (copied separately via tools/sync-models.sh)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
GENORBOX1_HOST="genorbox1"
GENORBOX1_USER="genorbox1"
PROD_BASE="/home/genorbox1/llama-server"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${YELLOW}==> $1${NC}"; }
ok() { echo -e "${GREEN}✓ $1${NC}"; }
err() { echo -e "${RED}✗ $1${NC}"; }

# Sanity check
if [[ ! -d server/inference_server ]]; then
    err "server/inference_server not found — run from ai-trainer root"
    exit 1
fi

# Sync inference_server source
log "Syncing inference_server → ${GENORBOX1_HOST}:${PROD_BASE}/inference_server/"
rsync -avz --delete \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='models' \
  --exclude='rag_data' \
  --exclude='*.gguf' \
  server/inference_server/ \
  "${GENORBOX1_USER}@${GENORBOX1_HOST}:${PROD_BASE}/inference_server/"
ok "inference_server synced"

# Sync finetune_studio source
log "Syncing finetune_studio → ${GENORBOX1_HOST}:${PROD_BASE}/finetune_studio/"
rsync -avz --delete \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='models' \
  --exclude='*.gguf' \
  trainer/src/finetune_studio/ \
  "${GENORBOX1_USER}@${GENORBOX1_HOST}:${PROD_BASE}/finetune_studio/"
ok "finetune_studio synced"

# Sync tests
log "Syncing tests → ${GENORBOX1_HOST}:${PROD_BASE}/tests/"
rsync -avz \
  --exclude='__pycache__' \
  --exclude='coverage_html' \
  --exclude='.pytest_cache' \
  tests/ "${GENORBOX1_USER}@${GENORBOX1_HOST}:${PROD_BASE}/tests/"
ok "tests synced"

# Verify on genorbox1
log "Running tests on genorbox1..."
ssh "${GENORBOX1_USER}@${GENORBOX1_HOST}" \
  "cd ${PROD_BASE} && find . -name __pycache__ -exec rm -rf {} + 2>/dev/null; python3 -m pytest tests/ -q 2>&1 | tail -3"

ok "Deploy complete"
echo ""
echo "Production status:"
echo "  curl -H 'Authorization: Bearer \$API_KEY' https://ai.smart-samurai.pl/health"
echo ""
echo "To restart llama-server on genorbox1:"
echo "  ssh ${GENORBOX1_HOST} 'sudo systemctl restart chris-ai.service'"