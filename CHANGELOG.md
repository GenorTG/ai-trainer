# CHANGELOG — AI Trainer

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 2026-08-20

### Added
- **ai-trainer repo**: Unified project structure (`trainer/` + `server/` + `data/` + `tests/`)
- **Git repo** with `.gitignore` excluding personal/runtime/model artifacts
- **Conventional Commits** setup
- **Pre-commit hooks** config (ruff, mypy, fast tests)
- **CONTRIBUTING.md** with workflow + commit conventions + file layout
- **README.md** with status, quickstart, architecture
- **3-layout path resolvers** in tests (genorbox1 + fan-dragon old + fan-dragon ai-trainer)

### Changed
- **Migrated** from `~/finetune-studio/` + `~/inference-server/` + `~/gemma-training/` to unified `~/ai-trainer/` repo
- **Renamed** `finetune-studio` → `trainer/` (kept Python package name `finetune_studio`)
- **Renamed** `inference-server` → `server/` (kept Python package name `inference_server`)

### Fixed
- **`finetune_studio.config.RAGSettings` forward-reference bug** — moved class def before Settings()
- **Starlette 1.0.0 `TemplateResponse` signature** — added required `request` first arg in pages.py
- **mocker.patch on unloaded modules** — added explicit imports before patch in conftest fixtures

## [0.1.0] — 2026-08-19

### Initial Setup
- **Chris AI v20** trained, exported (5.3GB GGUF), deployed to genorbox1
- **Chris AI v21** trained with knowledge preservation (97% persona + 3% general)
- **Tool-calling parser** handles 4+ formats (JSON, XML, Qwen native)
- **Tool-calling benchmark** at 90% (9/10) on v21 with native Jinja template
- **Jinja template extraction** from GGUF metadata (18808 chars for v21)
- **Industry-standard benchmarks** integrated (MMLU, HellaSwag, ARC, TruthfulQA, GSM8K, Winogrande)
- **Real benchmark results** vs Phi-4 14B baseline (v20: 49.2%, v21: 48.3%, Phi-4: 17.5%)
- **Test suite**: 573 tests passing (416 unit + 187 API + 105 frontend)
- **Coverage**: 64% overall (HTML report at `tests/coverage_html/`)
- **Lint clean**: ruff 0, pyflakes 0, compile 0, mypy 0
- **Cross-machine**: tests work on both genorbox1 and fan-dragon

### Added
- **finetune-studio**: full training app (WebUI + CLI + RAG + benchmarks + tool calling)
- **inference-server**: OpenAI-compatible API with RAG, MCP, samplers
- **DOCUMENTATION.md**: 13KB central reference, 63 files indexed
- **Dummy-proof comments**: line-by-line in renderer.py + inference.py (18KB each)
- **Teaching docstrings**: 63 files with "WHAT THIS FILE DOES" headers
- **DRY conftest.py**: 13KB shared fixtures, no duplication

### Known Issues
- **`agentic.py`** at 0% coverage — untested (next test target)
- **`parser.py`** at 0% coverage — legacy, may be removed
- **`tools.py`** at 13% coverage — needs more tests
- **`quality.py`** at 39% coverage — data tools API tests pending
- **Phi-4 14B baseline** blocked on ComfyUI GPU (15-17GB used)

[Unreleased]: https://github.com/internal/ai-trainer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/internal/ai-trainer/releases/tag/v0.1.0