# Changelog

All notable changes to AI Trainer are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- (nothing yet)

## [0.2.0] - 2026-08-20

### Added
- **Organizational infrastructure** for the unified `ai-trainer/` project:
  - `Makefile` with `test/lint/type-check/clean/deploy/pre-commit-install/version-bump-patch/all-checks/ci` targets
  - `.pre-commit-config.yaml` with ruff, mypy, pytest-fast, bandit, and local hygiene hooks
  - `pyproject.toml` shared config (ruff, mypy, pytest, coverage)
  - `tools/deploy.sh` — rsync inference_server + finetune_studio + tests to genorbox1
  - `tools/bump.py` — semver bump in both pyproject.toml files
  - `tools/check-secrets.sh` — pre-commit secret detection
  - `LICENSE` (MIT) with Gemma/Phi-4 third-party license notes
  - `SECURITY.md` — vulnerability reporting, security practices
  - `ARCHITECTURE.md` — ASCII system/data-flow/deployment diagrams
  - `API.md` — HTTP API reference for trainer + inference-server
  - `.editorconfig` — consistent coding style
  - `.gitattributes` — file type detection, export-ignore
  - `docs/DEVELOPMENT.md` — fan-dragon dev setup guide
  - `docs/TROUBLESHOOTING.md` — common issues + fixes
  - `docs/README.md` — central docs index
  - 3 ADRs in `docs/adr/`:
    - 001 — Why llama.cpp over vLLM
    - 002 — Why Gemma 4 E4B over Llama 3
    - 005 — Why fan-dragon as primary dev host
  - Per-module READMEs in `trainer/`, `server/`, `data/`, `tests/`, `models/`
- **GitHub Actions CI** (`.github/workflows/ci.yml`):
  - Lint job (ruff, bandit, pydocstyle)
  - Type-check job (mypy)
  - Test-unit job (fast unit tests)
  - Test-full job (all tests with coverage report)
  - Build job (Docker image)
- **153 new tests** across 4 files:
  - `test_tools.py` (32 tests) — `inference_server/tools.py` 13% → ~95%
  - `test_parser.py` (45 tests) — `inference_server/parser.py` 0% → ~95%
  - `test_agent.py` (46 tests) — `inference_server/agent.py` 13% → ~90%
  - `test_quality.py` (30 tests) — `finetune_studio/webui/routes/quality.py` 39% → ~100%
- **Cross-machine test path resolvers** — `tests/unit/test_config.py`, `tests/frontend/test_cli_gui_coverage.py`, `tests/frontend/test_pages.py`, `tests/frontend/test_static.py` updated to support 3 layouts:
  - genorbox1: `~/.openclaw/workspace/finetune-studio/`
  - fan-dragon (old): `~/finetune-studio/`
  - fan-dragon (new ai-trainer): `~/ai-trainer/trainer/`
- **Conventional Commits workflow** — `CONTRIBUTING.md` documents scopes and format

### Changed
- `trainer/src/finetune_studio/config.py` — `RAGSettings` moved before `Settings` class to avoid forward-reference NameError
- `trainer/src/finetune_studio/webui/routes/pages.py` — updated to Starlette 1.0.0 `TemplateResponse(request, name, context)` signature
- All test files: mock at source module for lazy imports
- `tests/frontend/conftest.py` — overrides `cleanup_caches` (tolerates missing parent modules)

### Fixed
- **Python scoping bug** in test path resolvers — list comprehensions no longer reference loop variables
- **`mocker.patch` failure** when module not loaded — explicit `import` before patch in 3 conftest.py files
- **ComfyUI interference** — documented rule: pause benchmark work + notify user when GPU >90%

### Statistics
- **Test count**: 573 → 726 tests passing
- **Source files**: 65 → 67 (added test files)
- **Documentation**: 5 → 15 markdown files
- **Git commits**: 7 conventional commits

## [0.1.0] - 2026-08-19

### Added
- **Initial release** of AI Trainer — unified AI training/benchmarking/hosting platform
- **trainer/** package (was `finetune-studio`):
  - FastAPI WebUI on port 7860
  - CLI with 15 commands (webui, train, test, suite, validate, convert, rag, compare, benchmark, analyze, augment, optimize, validate-hallucination, rag-test, models)
  - Training engine (QLoRA, Unsloth, TRL)
  - Industry-standard benchmarks (MMLU, HellaSwag, ARC, TruthfulQA, GSM8K, Winogrande)
  - 7 sampler presets (deterministic, balanced, creative, conservative, chris_ai_v20/v21, testing)
  - RAG with ChromaDB + sentence-transformers
  - MCP server with rag_search, rag_ingest, rag_list, calculator, web_search
  - Training quality tools (DataQualityAnalyzer, KnowledgePreserver, HallucinationGuard, DataAugmenter, ConfigOptimizer)
  - 33-format document parsers
  - Model comparison engine
- **server/** package (was `inference-server`):
  - v1 OpenAI-compatible server (port 8888)
  - v2 full-featured server (RAG + tool calling + MCP)
  - llama.cpp wrapper with full GPU offload
  - 33 document parsers
  - Jinja chat template extraction from GGUF
  - Tool-call parser (4+ formats: JSON, XML, Qwen native, function format)
  - 7 sampler presets
  - Real-time learning via `save_conversation_knowledge`
- **data/** scripts (training data generation):
  - `gen_v19.py`, `gen_v20.py`, `gen_v21.py` — synthetic persona data
  - `data_v21_training.jsonl` (6.1MB) — 97% persona
  - `data_v21_knowledge.jsonl` (18KB) — 3% general knowledge
- **tests/** suite:
  - 573 tests, 64% coverage
  - 27.7s end-to-end via master runner
  - DRY conftest.py with shared fixtures
  - All heavy resources mocked (GGUF, llama.cpp, GPU/CUDA, ComfyUI)

### Quality
- **Lint state**: ruff 0, pyflakes 0, compile 0, mypy 0 errors (across 65 source files)
- **Test state**: 573 passing, 2 skipped (fastapi TestClient, no real models)
- **Documentation**: 13KB DOCUMENTATION.md, per-file dummy-proof comments on critical modules

### Performance
- Chris AI v21 GGUF achieves **90% tool-calling** with native Jinja template
- Chris AI v20 (production): 49.2% overall on 6-benchmark suite
- Chris AI v21 (latest): 48.3% overall
- Phi-4 14B baseline: 17.5% (for comparison)

### Training
- v21 training: knowledge preservation via data mixing (97% persona + 3% general)
- v21 GGUF: 5.3GB, deployed to fan-dragon
- v20 GGUF: 5.0GB, deployed to genorbox1 production

[Unreleased]: https://github.com/genorbox1/ai-trainer/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/genorbox1/ai-trainer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/genorbox1/ai-trainer/releases/tag/v0.1.0