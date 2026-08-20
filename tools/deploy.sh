#!/usr/bin/env bash
# AI Trainer — Deploy to genorbox1 production
# Run from fan-dragon after commits.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
GENORBOX1_HOST="genorbox1"
GENORBOX1_USER="genorbox1"
PROD_DIR="/home/genorbox1/llama-server"
PROD_TEST_DIR="$PROD_DIR/tests"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}==> Syncing inference-server source${NC}"
rsync -avz --delete \
  --exclude='__pycache__' \
  --exclude='.venv' \
  --exclude='models' \
  --exclude='rag_data' \
  --exclude='*.gguf' \
  --exclude='debug_*.py' \
  server/ "${GENORBOX1_USER}@${GENORBOX1_HOST}:${PROD_DIR}/"

echo -e "${YELLOW}==> Syncing test suite${NC}"
rsync -avz \
  --exclude='__pycache__' \
  --exclude='coverage_html' \
  tests/ "${GENORBOX1_USER}@${GENORBOX1_HOST}:${PROD_TEST_DIR}/"

echo -e "${GREEN}==> Deploy complete${NC}"
echo ""
echo "Next steps on genorbox1:"
echo "  ssh ${GENORBOX1_HOST} 'cd ${PROD_DIR} && ./restart-llama.sh'"
echo ""
echo "Run tests:"
echo "  ssh ${GENORBOX1_HOST} 'cd ${PROD_DIR} && python -m pytest tests/ -q'"