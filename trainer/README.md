# Trainer (`trainer/`) — finetune-studio

Fine-tuning framework with WebUI + CLI + RAG + benchmarks + tool-calling.

## Quick Reference

| Command | Description |
|---------|-------------|
| `fts webui` | Start WebUI (port 7860) |
| `fts train` | Train a model |
| `fts test` | Test inference with a model |
| `fts suite` | Run test suite |
| `fts validate` | Validate training data |
| `fts convert` | Convert training data formats |
| `fts rag` | RAG operations |
| `fts compare` | Compare models |
| `fts benchmark` | Run benchmarks |
| `fts analyze` | Analyze data quality |
| `fts augment` | Augment training data |
| `fts optimize` | Optimize training config |
| `fts validate-hallucination` | Check for hallucinations |
| `fts rag-test` | Test RAG pipeline |
| `fts models` | List discovered models |

## Architecture

```
trainer/src/finetune_studio/
├── app.py                  # WebUI FastAPI entry
├── cli.py                  # CLI dispatcher
├── config.py               # Settings (host, port, model dirs)
├── benchmarks/             # Industry-standard benchmarks
│   ├── real_benchmarks.py  # MMLU, HellaSwag, ARC, etc.
│   ├── tool_calling.py     # Tool-call eval
│   ├── samplers.py         # 7 sampler presets
│   └── comparison.py       # Model comparison
├── compare/                # Cross-model comparison
├── data/                   # Data validation/dedup/conversion
├── models/                 # Model loader + registry
├── rag/                    # RAG store + ingest + query
├── testing/                # Inference engine for testing
├── training/               # Training engine + quality tools
└── webui/                  # FastAPI WebUI
    ├── app.py              # FastAPI app factory
    ├── routes/             # API endpoints
    ├── templates/          # Jinja HTML
    └── static/             # JS, CSS, images
```

## Development

```bash
# Install in dev mode
pip install -e ./trainer

# Run WebUI
fts webui

# Run tests
pytest tests/unit/test_training.py -v
pytest tests/api/test_finetune_studio.py -v
```

## Dependencies

- `fastapi>=0.115.0` — Web framework
- `uvicorn[standard]>=0.30.0` — ASGI server
- `torch>=2.1.0` — Training
- `transformers>=4.40.0` — HuggingFace models
- `trl>=0.12.0` — SFT training
- `peft>=0.14.0` — LoRA adapters
- `datasets>=3.0.0` — Data loading
- `accelerate>=1.0.0` — Multi-GPU
- `safetensors>=0.4.0` — Model I/O
- `jinja2>=3.1.0` — Templates
- `pydantic>=2.0.0` — Validation
- `rich>=13.0.0` — Terminal output

See `trainer/pyproject.toml` for full list.