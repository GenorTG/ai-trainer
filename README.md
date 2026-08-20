# AI Trainer — Trainer, Hoster, and Benchmarker for Chris AI

> A unified Python project for training, hosting, and benchmarking fine-tuned
> language models. Primary home: **fan-dragon** (RTX 3090, 24GB).
> Production runtime mirror: **genorbox1** (GTX 1070, 8GB).

## What It Does

| Component | Path | Purpose |
|-----------|------|---------|
| **Trainer** | `trainer/` | Fine-tune models with WebUI, CLI, RAG, benchmarks, tool-calling |
| **Server** | `server/` | OpenAI-compatible inference server with RAG, MCP, samplers |
| **Data** | `data/` | Training data generation scripts (persona + general knowledge) |
| **Tests** | `tests/` | DRY test suite (573 tests, 64% coverage, cross-machine) |

## Status

- ✅ **Chris AI v20** trained, exported, deployed to genorbox1, live in production
- ✅ **Chris AI v21** with knowledge preservation (97% persona + 3% general)
- ✅ **Tool-calling**: 90% (9/10) with native Gemma4 Jinja template
- ✅ **Benchmarks**: MMLU, HellaSwag, ARC, TruthfulQA, GSM8K, Winogrande
- ✅ **Production**: NPM proxy at `ai.smart-samurai.pl` → genorbox1 llama.cpp

## Quickstart

```bash
# Setup (fan-dragon, RTX 3090)
python3 -m venv ~/ai-trainer/.venv
source ~/ai-trainer/.venv/bin/activate
cd ~/ai-trainer/trainer && pip install -e .
cd ~/ai-trainer/server && pip install -e .
pip install pytest pytest-asyncio pytest-mock pytest-cov ruff mypy

# Run all checks
cd ~/ai-trainer
python tests/run_all.py --coverage

# Start the trainer WebUI
fts webui  # port 7860

# Start the inference server (with v21 GGUF)
inference-server serve --model models/gemma-4-e4b-it.Q4_K_M.gguf
```

## Architecture

```
ai-trainer/
├── trainer/                    # = old finetune-studio
│   ├── pyproject.toml           # 'finetune-studio' Python package
│   └── src/finetune_studio/    # WebUI + CLI + training engine
├── server/                     # = old inference-server
│   ├── pyproject.toml          # 'inference-server' Python package
│   └── inference_server/       # OpenAI-compatible API + RAG + MCP
├── data/                       # training data generation scripts
├── tests/                      # DRY test suite (573 tests)
│   ├── unit/                   # 416
│   ├── api/                    # 187
│   └── frontend/               # 105
├── docs/                       # per-module documentation
├── models/                     # gitignored model artifacts
└── tools/                      # build / lint / check scripts
```

## Development

```bash
# Run tests
python tests/run_all.py                        # all 573 tests
python tests/run_all.py --suite unit           # unit only
python tests/run_all.py --suite api            # API only
python tests/run_all.py --suite frontend       # frontend only
python tests/run_all.py --coverage             # with coverage report

# Lint
ruff check trainer/ server/                    # PEP 8 + extras
mypy trainer/src server/inference_server       # type check

# Pre-commit hooks (auto-fixes on commit)
pre-commit run --all-files
```

## Deployment

### Production (genorbox1)

```bash
# Pull latest on genorbox1
cd /home/genorbox1/llama-server
./sync-from-fan-dragon.sh                       # rsync scripts
./restart-llama.sh                              # bounce the service
```

### Development (fan-dragon)

```bash
# Just run from the source tree
source ~/ai-trainer/.venv/bin/activate
fts webui                                       # training WebUI
inference-server serve --model models/v21.gguf  # inference server
```

## Conventions
- [Conventional Commits](https://www.conventionalcommits.org/) for commit messages
- [Semantic Versioning](https://semver.org/) for releases
- Type hints everywhere (mypy strict-compatible)
- Test coverage target: ≥80% for new code

## Related Projects
- **Spice** (NSFW content agent) — separate, uses same inference server
- **Krzy-Kut Company Website** — consumer of inference server via `ai.smart-samurai.pl`

## License
Internal use only. Proprietary.

## Contact
- **Owner**: Chris (Master Genor) / Krzysztof Kutniowski
- **Maintainer**: Amy (OpenClaw primary agent)