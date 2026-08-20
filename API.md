# API Reference — Auto-generated docs index

This is a reference for HTTP API endpoints across both trainer and server packages.

## Trainer WebUI (`fts webui` on port 7860)

Base URL: `http://localhost:7860`

### Pages
| Path | Description |
|------|-------------|
| `/` | Dashboard |
| `/models` | Discovered models list |
| `/training` | Training controls |
| `/data` | Data management (validate, dedup, convert) |
| `/testing` | Model testing (chat, run-suite) |

### API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/models/list` | List all discovered models |
| GET | `/api/models/info` | Get model details |
| POST | `/api/models/refresh` | Re-scan model directories |
| GET | `/api/training/status` | Current training status |
| GET | `/api/training/progress` | Training progress (if running) |
| POST | `/api/training/start` | Start training |
| POST | `/api/training/stop` | Stop training |
| POST | `/api/compare/compare/load` | Load comparison models |
| POST | `/api/compare/compare/run` | Run comparison |
| POST | `/api/compare/compare/cleanup` | Cleanup comparison models |
| POST | `/api/compare/rag/chat` | RAG chat |
| POST | `/api/testing/load` | Load test model |
| POST | `/api/testing/unload` | Unload test model |
| GET | `/api/testing/status` | Testing status |
| POST | `/api/testing/chat` | Test chat |
| POST | `/api/testing/run-suite` | Run benchmark suite |
| GET | `/api/data/files` | List data files |
| POST | `/api/data/upload` | Upload data file |
| POST | `/api/data/validate` | Validate training data |
| GET | `/api/data/preview` | Preview data file |
| POST | `/api/data/dedup` | Remove duplicates |
| POST | `/api/data/convert` | Convert format |
| POST | `/api/data/analyze` | Analyze data quality |
| POST | `/api/data/augment` | Augment data |
| POST | `/api/data/optimize` | Optimize training config |
| POST | `/api/data/hallucination-check` | Check for hallucinations |

## Inference Server v2 (`inference-server` on port 8888)

Base URL: `http://localhost:8888`

### v1 OpenAI-compatible
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion (OpenAI-compatible) |
| GET | `/v1/models` | List loaded models |
| GET | `/health` | Health check |
| GET | `/stats` | Engine statistics |
| POST | `/reload` | Reload model |

### v2 Full-featured
| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completion (with `agentic: true` for tool calls) |
| GET | `/v1/models` | List loaded models |
| GET | `/v1/samplers` | List available sampler presets |
| POST | `/v1/rag/query` | Query RAG store |
| POST | `/v1/rag/ingest` | Ingest a document |
| POST | `/v1/rag/ingest-directory` | Ingest all files in a directory |
| GET | `/v1/rag/documents` | List ingested documents |
| DELETE | `/v1/rag/documents/{id}` | Remove a document |
| POST | `/v1/parse` | Parse tool calls from text |
| GET | `/v1/parse/supported` | List supported parser formats |
| GET | `/v1/mcp/tools` | List MCP tools |
| POST | `/v1/mcp/execute` | Execute MCP tool |

## Authentication

Inference server requires `Authorization: Bearer <API_KEY>` header if
`INFERENCE_API_KEY` env var is set. Trainer WebUI has no auth (local only).

## Rate Limits

No rate limits by default. Operator should add upstream rate limiting
(nginx, fail2ban, etc.) if exposing publicly.

## Examples

### Chat completion (OpenAI-compatible)
```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $INFERENCE_API_KEY" \
  -d '{
    "model": "chris-ai-v21",
    "messages": [{"role": "user", "content": "Cześć!"}],
    "temperature": 0.45,
    "top_p": 0.9,
    "top_k": 30,
    "repeat_penalty": 1.02,
    "min_p": 0.02
  }'
```

### Tool calling (v2 agentic)
```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $INFERENCE_API_KEY" \
  -d '{
    "model": "chris-ai-v21",
    "messages": [{"role": "user", "content": "Search RAG for finetune notes"}],
    "agentic": true
  }'
```

### RAG ingest
```bash
curl -X POST http://localhost:8888/v1/rag/ingest \
  -H "Authorization: Bearer $INFERENCE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/doc.md",
    "chunk_size": 512,
    "chunk_overlap": 50
  }'
```

### MCP tool execution
```bash
curl -X POST http://localhost:8888/v1/mcp/execute \
  -H "Authorization: Bearer $INFERENCE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "calculator",
    "arguments": {"expression": "sqrt(144) + 5"}
  }'
```

## See Also

- [server/README.md](../server/README.md) — server architecture details
- [trainer/README.md](../trainer/README.md) — trainer CLI + WebUI details
- [DOCUMENTATION.md](../DOCUMENTATION.md) — codebase-wide reference