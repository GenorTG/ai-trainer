# AI Trainer — Documentation Index

This is the canonical entry point for all documentation.

## Quick Links

| Doc | Purpose |
|-----|---------|
| [Root README](../README.md) | Project overview, quickstart, architecture |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Workflow, commit conventions, coding standards |
| [CHANGELOG.md](../CHANGELOG.md) | Version history |

## Per-Module Documentation

| Module | Doc | Purpose |
|--------|-----|---------|
| `trainer/` | [README](../trainer/README.md) | Training framework docs |
| `server/` | [README](../server/README.md) | Inference server docs |
| `data/` | [README](../data/README.md) | Training data scripts |
| `tests/` | [README](../tests/README.md) | Test suite docs |
| `models/` | [README](../models/README.md) | Model artifacts |

## Architecture Decision Records (ADRs)

This folder will contain ADRs documenting major decisions:
- ADR-001: Why llama.cpp over vLLM for inference
- ADR-002: Why Gemma 4 over Llama 3 for Chris AI
- ADR-003: Why data mixing (97% persona + 3% general) for v21
- ADR-004: Why native Jinja template extraction over hardcoded format
- ADR-005: Why fan-dragon as primary development host

## API Reference

### Trainer CLI (`fts`)

| Command | Description |
|---------|-------------|
| `fts webui` | Start WebUI |
| `fts train` | Train a model |
| `fts test` | Test inference |
| `fts suite` | Run test suite |
| `fts validate` | Validate training data |
| `fts convert` | Convert formats |
| `fts rag` | RAG operations |
| `fts compare` | Compare models |
| `fts benchmark` | Run benchmarks |
| `fts analyze` | Analyze data quality |
| `fts augment` | Augment training data |
| `fts optimize` | Optimize config |
| `fts validate-hallucination` | Check hallucinations |
| `fts rag-test` | Test RAG pipeline |
| `fts models` | List discovered models |

### Server CLI (`inference-server`)

| Command | Description |
|---------|-------------|
| `inference-server serve` | Start chat server (OpenAI-compatible) |
| `inference-server start` | Start FastAPI server |
| `inference-server rag` | RAG CLI |
| `inference-server parse` | Parse tool calls |

### HTTP API

See [server/README.md](../server/README.md#api-endpoints) for full HTTP API.

## Performance & Benchmarks

### Chris AI v20 (current production)
- MMLU: 35% | HellaSwag: 55% | ARC: 85%
- TruthfulQA: 0% (by design — persona restricted)
- GSM8K: 50% | Winogrande: 70%
- **Overall: 49.2%** on 20 samples / 8K context

### Chris AI v21 (latest)
- MMLU: 35% | HellaSwag: 45% | ARC: 85%
- TruthfulQA: 0% (by design)
- GSM8K: 50% | Winogrande: 75%
- **Overall: 48.3%** on 20 samples / 8K context
- **Tool calling: 90% (9/10)** with native Jinja template

### Phi-4 14B baseline (MIT)
- MMLU: 0% | HellaSwag: 75% | ARC: 20%
- GSM8K: 40% | Winogrande: 40%
- **Overall: 17.5%** on same benchmarks

### Test Suite
- **573 tests passing, 2 skipped**
- 64% overall coverage
- 27.7s end-to-end via `python tests/run_all.py`