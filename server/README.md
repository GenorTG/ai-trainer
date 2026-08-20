# Inference Server with RAG

Portable, self-contained inference server with RAG — pull from GitHub, point to model + docs, run.

## Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone
git clone .../inference-server
cd inference-server

# Add model
cp /path/to/model.gguf models/

# Add documents
cp -r /path/to/docs/* rag_data/documents/

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your settings

# Run
docker compose up -d
```

### Option 2: Python
```bash
# Install
pip install -e ".[all]"

# Ingest documents
inference-server --ingest /path/to/docs/

# Run
inference-server --config config.yaml
```

## API

### OpenAI-Compatible
```bash
# Chat
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What projects?"}]}'

# List models
curl http://localhost:8080/v1/models
```

### RAG-Enhanced
```bash
# Query with RAG context
curl http://localhost:8080/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What projects?", "top_k": 5}'

# Ingest new document
curl -X POST http://localhost:8080/v1/rag/ingest \
  -F "file=@document.pdf"

# List indexed documents
curl http://localhost:8080/v1/rag/documents
```

### Management
```bash
# Health check
curl http://localhost:8080/health

# Statistics
curl http://localhost:8080/stats

# Reload model
curl -X POST http://localhost:8080/reload
```

## Configuration

See `config.example.yaml` for all options.

### Environment Variables
- `MODEL_PATH` — Override model path
- `RAG_STORE_PATH` — Override RAG store path
- `API_KEY` — Set API key
- `PORT` — Override port

## What You Need

### Hardware
- GPU: RTX 3090 24GB (recommended) or RTX 3080 16GB+
- RAM: 32GB+ (64GB for large models)
- Storage: 50GB+ (model + embeddings + docs)

### Software
- Python 3.12/3.13 (for Python install)
- Docker + NVIDIA Container Toolkit (for Docker install)
- CUDA 12.x

### Model
- GGUF format (recommended — single file, portable)
- Or safetensors directory

### Documents
- PDF, DOCX, TXT, MD, CSV, JSON, code files
- Project docs, terrain data, specs, etc.

## Deployment

### Local Development
```bash
inference-server --model ./model.gguf --rag-dir ./docs
```

### Production (Docker)
```bash
docker compose up -d
# Model + RAG data persist in volumes
```

### Cloud (AWS/GCP/Azure)
```bash
# Same Docker setup, just point volumes to cloud storage
# Or use the Python install with systemd
```

## Updating Documents

No retraining needed — just ingest new documents:

```bash
# Via CLI
inference-server --ingest /path/to/new-docs/

# Via API
curl -X POST http://localhost:8080/v1/rag/ingest \
  -F "file=@new_doc.pdf"
```

## Architecture

```
inference-server/
├── config.yaml          # Configuration
├── models/              # Model files (GGUF/safetensors)
├── rag_data/
│   ├── documents/       # Source documents
│   └── store/           # ChromaDB vector store
├── src/
│   ├── server.py        # FastAPI server
│   ├── inference.py     # Model inference
│   ├── rag.py           # RAG + document ingestion
│   ├── config.py        # Configuration
│   └── cli.py           # CLI interface
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```
