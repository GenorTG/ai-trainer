# Development Setup — fan-dragon

If you're forking this repo, here's how to set up fan-dragon as your dev host.

## Prerequisites

- Linux with systemd (Arch, Fedora, Ubuntu, etc.)
- NVIDIA GPU with CUDA 12+ (RTX 3090 recommended)
- 32GB+ system RAM
- 100GB+ free disk (for models)
- Python 3.10+
- fish or bash shell

## 1. Install System Dependencies

### Arch Linux
```bash
sudo pacman -S python python-pip git cuda cudnn
yay -S heroic-launcher  # optional, for benchmarking
```

### Ubuntu / Debian
```bash
sudo apt install python3 python3-pip git
# CUDA: https://developer.nvidia.com/cuda-downloads
```

## 2. Clone Repository

```bash
git clone https://github.com/genorbox1/ai-trainer.git ~/ai-trainer
cd ~/ai-trainer
```

## 3. Set Up Virtual Environments

We use two separate venvs because trainer (training) and server (inference)
have conflicting dependency versions.

```bash
# Trainer venv (torch, transformers, trl, peft)
python -m venv trainer/.venv
source trainer/.venv/bin/activate
pip install -U pip
pip install -e .

# Server venv (llama-cpp-python, fastapi)
python -m venv server/.venv
source server/.venv/bin/activate
pip install -U pip
pip install -e .
```

## 4. Install Dev Tools

```bash
pip install pre-commit ruff mypy bandit pydocstyle pytest pytest-cov
pre-commit install
```

## 5. Verify Installation

```bash
# Trainer
source trainer/.venv/bin/activate
fts --help

# Server
source server/.venv/bin/activate
inference-server --help

# Tests
cd ~/ai-trainer
python tests/run_all.py
```

## 6. Configure Git

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
git config commit.gpgsign false  # or true if you have GPG
```

## 7. Set Up SSH Access

For Tailscale access between machines:
```bash
sudo pacman -S tailscale
sudo systemctl enable --now tailscaled
sudo tailscale up
```

## Common Issues

### CUDA out of memory
- Use QLoRA instead of full fine-tuning
- Reduce batch size: `--per-device-train-batch-size 1`
- Use gradient accumulation: `--gradient-accumulation-steps 8`

### llama-cpp-python fails to build
```bash
pip install llama-cpp-python --config-settings='cmake.args="-DGGML_CUDA=on"'
```

### ComfyUI using all GPU
- Pause benchmarks, wait for user to free GPU
- See `docs/adr/005-fan-dragon-primary.md` for the rule

## Cross-Machine Development

If you also have a production server (e.g., genorbox1):
```bash
# Deploy from fan-dragon → production
tools/deploy.sh

# Verify on production
ssh production 'cd /home/prod/llama-server && python -m pytest tests/ -q'
```

## IDE Setup

### VS Code
Install extensions:
- Python
- Pylance
- Ruff
- Even Better TOML
- GitLens

### PyCharm
Configure interpreter to use `trainer/.venv/bin/python` or `server/.venv/bin/python`.

## See Also

- [README.md](../README.md) — project overview
- [docs/adr/001-llama-cpp.md](../docs/adr/001-llama-cpp.md) — why llama.cpp
- [docs/adr/005-fan-dragon-primary.md](../docs/adr/005-fan-dragon-primary.md) — fan-dragon architecture