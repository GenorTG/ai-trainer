#!/usr/bin/env bash
set -euo pipefail
echo "=== Finetune Studio Installer ==="

# Python version selection:
#   Set PYTHON_VERSION=3.12 or PYTHON_VERSION=3.13 to force a version
#   Otherwise auto-detects 3.13 (preferred) or 3.12

PYTHON_CMD=""
PYTHON_VER=""

if [ -n "${PYTHON_VERSION:-}" ]; then
    # User specified a version
    for py in "python${PYTHON_VERSION}" python3; do
        if command -v "$py" &>/dev/null; then
            ver=$("$py" --version 2>&1 | grep -oP '\d+\.\d+')
            if [ "$ver" = "$PYTHON_VERSION" ]; then
                PYTHON_CMD="$py"
                PYTHON_VER="$ver"
                break
            fi
        fi
    done
    if [ -z "$PYTHON_CMD" ]; then
        echo "ERROR: Python $PYTHON_VERSION not found."
        echo "Install Python $PYTHON_VERSION from https://www.python.org/downloads/"
        exit 1
    fi
else
    # Auto-detect (prefer 3.13, fall back to 3.12)
    for py in python3.13 python3.12 python3; do
        if command -v "$py" &>/dev/null; then
            ver=$("$py" --version 2>&1 | grep -oP '\d+\.\d+')
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "$major" -ge 3 ] && [ "$minor" -ge 12 ]; then
                PYTHON_CMD="$py"
                PYTHON_VER="$ver"
                break
            fi
        fi
    done
    if [ -z "$PYTHON_CMD" ]; then
        echo "ERROR: Python 3.12+ not found."
        echo "Install Python 3.12 or 3.13 from https://www.python.org/downloads/"
        exit 1
    fi
fi

echo "Python: $PYTHON_CMD ($PYTHON_VER)"

# Install UV if missing
if ! command -v uv &>/dev/null; then
    echo "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "UV: $(uv --version)"

# Create venv with selected Python
echo "Creating venv with Python $PYTHON_VER..."
rm -rf .venv
uv venv .venv --python "$PYTHON_CMD"

# Install packages
echo "Installing packages..."
source .venv/bin/activate
uv pip install -e "."

# Generate lock file for reproducibility
if [ ! -f "uv.lock" ]; then
    echo "Generating lock file..."
    uv lock
fi

mkdir -p data
echo ""
echo "=== Install complete! ==="
echo "Python: $(python --version)"
echo "Run: bash run.sh"
