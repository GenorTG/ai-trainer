# 🧠 AI Trainer

Fine-tuning, inference, and benchmarking toolkit for Gemma/Phi language models.

[![CI](https://github.com/GenorTG/ai-trainer/actions/workflows/ci.yml/badge.svg)](https://github.com/GenorTG/ai-trainer/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Fine-tuning Studio** — WebUI + CLI for training Gemma/Phi models with LoRA/QLoRA
- **Inference Server** — OpenAI-compatible API with RAG, tool calling, and MCP support
- **Benchmark Suite** — Real HuggingFace datasets for tool calling, persona, and reasoning
- **Knowledge Preservation** — 97/3 data mixing to prevent catastrophic forgetting
- **Native Jinja Templates** — Model-specific tool-call formatting for 90%+ accuracy

## Quick Start

```bash
git clone https://github.com/GenorTG/ai-trainer.git
cd ai-trainer

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install packages
pip install -e trainer/ server/

# Run tests
make test

# Start inference server
cd server
llama-server -m ../models/your-model.gguf --host 0.0.0.0 --port 8080 -ngl 99 -c 8192 --jinja
python -m inference_server.server  # OpenAI-compatible proxy on port 5000
```

## Project Structure

```
ai-trainer/
├── trainer/              # Fine-tuning studio (WebUI + CLI)
│   └── src/finetune_studio/
│       ├── training/     # Engine, data quality, config optimizer
│       ├── benchmarks/   # Tool calling, scoring, samplers
│       ├── compare/      # Model comparison engine
│       ├── rag/          # Retrieval-augmented generation
│       └── webui/        # FastAPI WebUI
├── server/               # Inference server
│   └── inference_server/
│       ├── templates/    # Native Jinja template renderer
│       ├── tools.py      # Tool definitions + RAG tools
│       ├── parser.py     # Multi-format tool-call parser
│       └── agent.py      # Agentic loop with tool execution
├── tests/                # 806 tests, 60% coverage
├── docs/                 # ADRs, development guide, troubleshooting
└── tools/                # deploy.sh, bump.py, check-secrets.sh
```

## Branches

| Branch | Purpose | Protection |
|--------|---------|------------|
| `main` | Stable releases | PR required, stale reviews dismissed |
| `dev` | Active development | Direct push |

## Development

```bash
# Install dev dependencies
pip install -e "trainer/[dev]" -e "server/[dev]"

# Run full test suite
make test

# Run linter
make lint

# Run type checker
make type-check

# All checks at once
make all-checks
```

## Documentation

- [Architecture](ARCHITECTURE.md) — System diagrams and data flow
- [API Reference](API.md) — Full HTTP API documentation
- [Development Guide](docs/DEVELOPMENT.md) — Setup and workflow
- [Troubleshooting](docs/TROUBLESHOOTING.md) — Common issues and fixes
- [Architecture Decision Records](docs/adr/) — Key design decisions

## License

MIT — see [LICENSE](LICENSE) for details.

Gemma models are subject to [Google's Terms of Use](https://ai.google.dev/gemma/terms).
Phi-4 is licensed under [MIT](https://huggingface.co/microsoft/phi-4/blob/main/LICENSE).
