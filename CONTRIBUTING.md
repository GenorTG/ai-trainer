# Contributing to AI Trainer

## Workflow

1. **Branch**: Create a feature branch (`feat/...`, `fix/...`, `chore/...`)
2. **Develop**: Write code + tests together (TDD preferred)
3. **Test**: Run `python tests/run_all.py --coverage`
4. **Lint**: Run `ruff check trainer/ server/`
5. **Type**: Run `mypy trainer/src server/inference_server`
6. **Commit**: Use [Conventional Commits](https://www.conventionalcommits.org/)
7. **Push**: `git push origin feature/my-...`
8. **PR**: Open PR with description, link to issue if any

## Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type | Use for |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `chore` | Maintenance (deps, configs, scripts) |
| `docs` | Documentation only |
| `style` | Formatting (no logic change) |
| `refactor` | Code change (no feature/fix) |
| `test` | Adding/updating tests |
| `perf` | Performance improvement |
| `build` | Build system / dependencies |

### Scopes (for this project)

| Scope | Area |
|-------|------|
| `trainer` | `trainer/` package (finetune-studio) |
| `server` | `server/` package (inference-server) |
| `tests` | `tests/` suite |
| `data` | `data/` training scripts |
| `docs` | `docs/` per-module docs |
| `tools` | `tools/` scripts |
| `ci` | CI / pre-commit hooks |

### Examples

```
feat(trainer): add tool-calling parser for Qwen native format

Implements parser for `<|tool_call>call:name{args}<tool_call|>` format
with `<|"|">` arg delimiters. Achieves 90% (9/10) on v21 GGUF.

Closes #42

fix(server): handle empty body in MCP execute endpoint

Previously raised JSONDecodeError when body was missing.
Now returns 400 with clear error message.

test(tests): add coverage for agentic.py module

Was at 0% coverage. Added 47 unit tests + 12 API tests.
Brings overall coverage to 64% → 71%.
```

## Coding Standards

### Style
- **PEP 8** with line length 100 (configured in `pyproject.toml`)
- **Type hints** required on all new code
- **Docstrings** required on all public functions/classes
- **No `chr()` hacks** — use string literals
- **No `sys.path.insert(0, ...)`** — proper package installation
- **No duplicate code** — DRY principle

### Tests
- **Every PR** must include tests for new functionality
- **Coverage target**: ≥80% for new code, ≥64% overall (current)
- **No real I/O in tests** — mock GGUF, llama.cpp, GPU, network
- **Use fixtures from `tests/conftest.py`** — don't redefine

### Type Checking
- `mypy` is enforced via pre-commit hook
- Per-instance `# type: ignore` ONLY for 100% sure false positives
- `# mypy: disable-error-code="X,Y"` at file top for whole-file suppressions

## File Layout

```
trainer/                          # 'finetune-studio' package
├── pyproject.toml                 # dependency config
├── cli.py                         # CLI entry (fts command)
├── run_*.py                       # benchmark runners
├── data_v*.jsonl                  # training datasets (tracked)
└── src/finetune_studio/           # Python package
    ├── app.py                     # WebUI FastAPI
    ├── cli.py                     # CLI dispatcher
    ├── config.py                  # Settings + RAGSettings
    ├── benchmarks/                # MMLU, HellaSwag, ARC, etc.
    ├── compare/                   # Model comparison
    ├── data/                      # Data validation/dedup/conversion
    ├── models/                    # Model loader + registry
    ├── rag/                       # RAG store + ingest + query
    ├── testing/                   # Inference engine for testing
    ├── training/                  # Training engine + quality tools
    └── webui/                     # FastAPI WebUI + routes

server/                           # 'inference-server' package
├── pyproject.toml                 # dependency config
├── config.yaml                    # runtime config
└── inference_server/              # Python package
    ├── server.py                  # v1 OpenAI-compatible
    ├── server_v2.py               # v2 (samplers + agentic)
    ├── inference.py               # llama.cpp wrapper
    ├── rag.py                     # RAG store + ingest
    ├── mcp.py                     # MCP server
    ├── templates/                 # Jinja chat templates
    └── ...

tests/                            # DRY test suite (573 tests)
├── conftest.py                    # root fixtures
├── unit/                          # 416 unit tests
├── api/                           # 187 API tests
├── frontend/                      # 105 frontend tests
├── run_all.py                     # master runner
└── README.md                      # test docs
```

## Pre-commit Hooks

```bash
# Install once
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

Hooks run on commit:
- `ruff check --fix` — auto-fix lint issues
- `ruff format` — auto-format
- `mypy` — type check
- `pytest tests/unit -q` — fast unit tests (must pass)
- `trailing-whitespace`, `end-of-file-fixer` — file hygiene

Hooks NOT run on commit (slow):
- `pytest tests/api` — API tests
- `pytest tests/frontend` — frontend tests
- `pytest tests/ --cov` — full suite with coverage

## Release Process

1. Bump version: `python -c "from trainer import __version__; print(__version__)"`
2. Update CHANGELOG.md
3. Tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
4. Push tag: `git push origin v0.1.0`
5. CI builds wheels, deploys to genorbox1

## Questions?

Open an issue or ping Amy in Discord.