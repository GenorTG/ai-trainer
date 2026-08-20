# Inference Server with RAG — Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [What You Need](#what-you-need)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Server](#running-the-server)
6. [API Reference](#api-reference)
7. [RAG (Retrieval-Augmented Generation)](#rag)
8. [Model Management](#model-management)
9. [Docker Deployment](#docker-deployment)
10. [Troubleshooting](#troubleshooting)
11. [Architecture](#architecture)
12. [Cost Analysis](#cost-analysis)

---

## Overview

This is a **portable, self-contained inference server** with built-in RAG (Retrieval-Augmented Generation). It's designed to be:

- **Easy to deploy** — pull from GitHub, point to model + docs, run
- **OpenAI-compatible** — drop-in replacement for existing integrations
- **RAG-enabled** — query documents without retraining
- **Persistent** — vector store survives restarts
- **No internet needed** — fully self-contained

### What It Does

1. **Loads a local model** (GGUF or safetensors)
2. **Serves an OpenAI-compatible API** at `http://localhost:8080`
3. **Indexes documents** into a vector store (ChromaDB)
4. **Retrieves relevant context** when answering questions
5. **Generates responses** using the local model + RAG context

---

## What You Need

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 3080 16GB | RTX 3090 24GB |
| **RAM** | 32GB | 64GB |
| **Storage** | 50GB | 100GB+ |
| **CPU** | 4 cores | 8+ cores |

### Software

- **Python** 3.12 or 3.13 (or Docker)
- **CUDA** 12.x (for GPU inference)
- **Git** (to clone the repo)

### Model

- **GGUF format** (recommended — single file, portable)
  - Download from HuggingFace: `TheBloke/Llama-2-7B-GGUF` or similar
  - Or use your own trained model
- **Safetensors format** (HuggingFace directory)
  - Any model from HuggingFace Hub

### Documents for RAG

- **Supported formats**: PDF, DOCX, TXT, MD, CSV, JSON, JSONL, Python, JavaScript, TypeScript, HTML, CSS
- **Examples**: Project documentation, terrain data, company specs, policies, FAQs

---

## Installation

### Option 1: Python (Development/Testing)

```bash
# 1. Clone the repo
git clone https://github.com/your-org/inference-server.git
cd inference-server

# 2. Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .\.venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -e ".[all]"

# 4. Verify installation
inference-server --help
```

### Option 2: Docker (Production)

```bash
# 1. Clone the repo
git clone https://github.com/your-org/inference-server.git
cd inference-server

# 2. Build the image
docker compose build

# 3. Run
docker compose up -d
```

### Option 3: Quick Install (No Git)

```bash
# 1. Download the release
wget https://github.com/your-org/inference-server/releases/latest/download/inference-server.tar.gz
tar -xzf inference-server.tar.gz
cd inference-server

# 2. Install
pip install -e ".[all]"
```

---

## Configuration

### Config File (config.yaml)

```yaml
server:
  host: "0.0.0.0"      # Bind address
  port: 8080            # Bind port

model:
  path: "./models/model.gguf"  # Path to model file/directory
  n_gpu_layers: 99             # GPU layers to offload (99 = all)
  n_ctx: 8192                  # Context window size
  n_threads: 4                 # CPU threads

rag:
  enabled: true                # Enable/disable RAG
  store_path: "./rag_data/store"      # Vector store location
  documents_path: "./rag_data/documents"  # Where to find documents
  embedding_model: "all-MiniLM-L6-v2"    # Embedding model
  chunk_size: 512              # Words per chunk
  chunk_overlap: 50            # Overlap between chunks
  top_k: 5                     # Number of chunks to retrieve
  min_score: 0.3               # Minimum similarity score

inference:
  max_tokens: 1024             # Max response length
  temperature: 0.7             # Sampling temperature
  top_p: 0.9                   # Nucleus sampling
  repeat_penalty: 1.05         # Repetition penalty

api:
  key: ""                      # API key (empty = no auth)
  cors: true                   # Enable CORS
```

### Environment Variable Overrides

| Variable | Overrides |
|----------|-----------|
| `MODEL_PATH` | `model.path` |
| `RAG_STORE_PATH` | `rag.store_path` |
| `API_KEY` | `api.key` |
| `PORT` | `server.port` |

### Quick Start Config

```bash
# 1. Copy example config
cp config.example.yaml config.yaml

# 2. Edit with your settings
nano config.yaml

# 3. Or use environment variables
export MODEL_PATH=/path/to/model.gguf
export PORT=8080
```

---

## Running the Server

### Python

```bash
# Basic run
inference-server --config config.yaml

# Override model path
inference-server --model /path/to/model.gguf

# Override port
inference-server --port 9090

# Disable RAG
inference-server --no-rag

# Ingest documents only (no server)
inference-server --ingest /path/to/documents/
```

### Docker

```bash
# Start
docker compose up -d

# Stop
docker compose down

# View logs
docker compose logs -f

# Restart
docker compose restart
```

### Systemd (Linux Production)

```ini
# /etc/systemd/system/inference-server.service
[Unit]
Description=Inference Server with RAG
After=network.target

[Service]
Type=simple
User=inference
WorkingDirectory=/opt/inference-server
ExecStart=/opt/inference-server/.venv/bin/inference-server --config /opt/inference-server/config.yaml
Restart=always
RestartSec=10
Environment=CUDA_VISIBLE_DEVICES=0

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable inference-server
sudo systemctl start inference-server
```

---

## API Reference

### Authentication

If `api.key` is set in config, all requests must include:

```
Authorization: Bearer YOUR_API_KEY
```

### OpenAI-Compatible Endpoints

#### POST /v1/chat/completions

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What projects has the company completed?"}
    ],
    "max_tokens": 1024,
    "temperature": 0.7
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "default",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The company has completed 15 commercial projects..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

#### GET /v1/models

```bash
curl http://localhost:8080/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "default",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local"
    }
  ]
}
```

### RAG Endpoints

#### POST /v1/rag/query

Query with RAG context (retrieves relevant documents, injects into prompt):

```bash
curl http://localhost:8080/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the budget for Project X?",
    "top_k": 5,
    "max_tokens": 1024
  }'
```

**Response:**
```json
{
  "response": "Based on the documents, the budget for Project X is...",
  "sources": [
    {
      "text": "Budget document snippet...",
      "score": 0.85,
      "source": "rag_data/documents/budget_x.pdf"
    }
  ],
  "chunks_retrieved": 3,
  "time_ms": 1250.5
}
```

#### POST /v1/rag/ingest

Upload a document to the RAG store:

```bash
curl -X POST http://localhost:8080/v1/rag/ingest \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "file": "rag_data/documents/document.pdf",
  "document_id": "abc123def456",
  "chunks": 15
}
```

#### POST /v1/rag/ingest-directory

Ingest all documents from a directory:

```bash
curl -X POST http://localhost:8080/v1/rag/ingest-directory \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/documents"}'
```

#### GET /v1/rag/documents

List all indexed documents:

```bash
curl http://localhost:8080/v1/rag/documents
```

**Response:**
```json
{
  "documents": [
    {
      "document_id": "abc123def456",
      "chunk_count": 15,
      "sources": ["rag_data/documents/report.pdf"]
    }
  ],
  "total_chunks": 150
}
```

#### DELETE /v1/rag/documents/{document_id}

Remove a document from the RAG store:

```bash
curl -X DELETE http://localhost:8080/v1/rag/documents/abc123def456
```

### Management Endpoints

#### GET /health

```bash
curl http://localhost:8080/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "rag_enabled": true,
  "uptime_seconds": 3600.5
}
```

#### GET /stats

```bash
curl http://localhost:8080/stats
```

**Response:**
```json
{
  "model": {
    "loaded": true,
    "model_path": "./models/model.gguf",
    "is_gguf": true
  },
  "rag": {
    "enabled": true,
    "total_chunks": 150,
    "documents": 12
  },
  "uptime_seconds": 3600.5
}
```

#### POST /reload

Reload model and RAG from config:

```bash
curl -X POST http://localhost:8080/reload
```

---

## RAG

### How RAG Works

1. **Document Ingestion**: Documents are split into chunks (512 words each)
2. **Embedding**: Each chunk is converted to a vector using sentence-transformers
3. **Storage**: Vectors are stored in ChromaDB (persistent)
4. **Query**: User question is embedded, similar chunks are retrieved
5. **Context Injection**: Retrieved chunks are added to the system prompt
6. **Generation**: Model generates response using the augmented prompt

### Ingesting Documents

#### Via CLI (Before Server Start)

```bash
# Ingest a directory
inference-server --ingest /path/to/documents/

# Ingest specific file types only
inference-server --ingest /path/to/docs --extensions .pdf,.docx,.txt
```

#### Via API (While Server Running)

```bash
# Upload single file
curl -X POST http://localhost:8080/v1/rag/ingest \
  -F "file=@new_document.pdf"

# Ingest directory
curl -X POST http://localhost:8080/v1/rag/ingest-directory \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/new-docs"}'
```

### Updating Documents

**No retraining needed!** Just ingest new documents:

```bash
# Add new document
curl -X POST http://localhost:8080/v1/rag/ingest \
  -F "file=@updated_project.pdf"

# Remove old document
curl -X DELETE http://localhost:8080/v1/rag/documents/old_doc_id
```

### RAG Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 512 | Words per chunk |
| `chunk_overlap` | 50 | Overlap between chunks |
| `top_k` | 5 | Number of chunks to retrieve |
| `min_score` | 0.3 | Minimum similarity score |
| `embedding_model` | all-MiniLM-L6-v2 | Embedding model |

---

## Model Management

### Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| **GGUF** | `.gguf` | Recommended — single file, portable |
| **Safetensors** | directory | HuggingFace format |

### Downloading Models

```bash
# From HuggingFace (GGUF)
wget -O models/model.gguf \
  https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

# From HuggingFace (safetensors)
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf models/llama-2-7b-chat-hf
```

### Using Your Trained Model

```bash
# 1. Export from Finetune Studio
fts train /path/to/base-model /path/to/data.jsonl --output ./my-model

# 2. Convert to GGUF (if needed)
python -m llama_cpp.llama_export --outtype q4_k_m --outfile my-model.gguf ./my-model/adapter

# 3. Place in models directory
cp my-model.gguf models/

# 4. Update config
# model.path: "./models/my-model.gguf"
```

### Model Selection Guide

| Model Size | VRAM Required | Use Case |
|------------|---------------|----------|
| 7B Q4 | ~5GB | Light tasks, chat |
| 13B Q4 | ~9GB | Balanced |
| 30B Q4 | ~20GB | Complex reasoning |
| 70B Q4 | ~40GB | Maximum quality |

---

## Docker Deployment

### docker-compose.yml

```yaml
version: "3.8"

services:
  inference-server:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./models:/app/models           # Model files
      - ./rag_data:/app/rag_data       # RAG data + store
      - ./config.yaml:/app/config.yaml # Configuration
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

### Commands

```bash
# Build
docker compose build

# Start (detached)
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down

# Restart
docker compose restart

# Shell into container
docker compose exec inference-server bash
```

### Docker with Custom Config

```bash
# 1. Create config
cp config.example.yaml config.yaml
nano config.yaml

# 2. Run with custom config
docker compose up -d

# Or override via environment
MODEL_PATH=/app/models/model.gguf PORT=9090 docker compose up -d
```

---

## Troubleshooting

### Common Issues

#### "No model loaded"

**Cause**: Model path is wrong or model file doesn't exist.

**Fix**:
```bash
# Check model exists
ls -la models/

# Check config
cat config.yaml | grep path

# Test model loading
python -c "from llama_cpp import Llama; m = Llama(model_path='models/model.gguf'); print('OK')"
```

#### "RAG not enabled"

**Cause**: RAG is disabled in config or ChromaDB failed to initialize.

**Fix**:
```bash
# Check config
cat config.yaml | grep -A5 rag

# Test ChromaDB
python -c "import chromadb; c = chromadb.PersistentClient(path='rag_data/store'); print('OK')"
```

#### "CUDA out of memory"

**Cause**: Model too large for GPU VRAM.

**Fix**:
```bash
# Reduce GPU layers
# In config.yaml:
# model.n_gpu_layers: 50  # Instead of 99

# Or use a smaller model
# Q4_K_M instead of Q8_K_M
```

#### "Connection refused"

**Cause**: Server not running or wrong port.

**Fix**:
```bash
# Check if server is running
curl http://localhost:8080/health

# Check port
netstat -tlnp | grep 8080

# Check logs
docker compose logs inference-server
```

#### "Import error: No module named 'llama_cpp'"

**Cause**: llama-cpp-python not installed.

**Fix**:
```bash
pip install llama-cpp-python
# or for CUDA support:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

### Performance Tuning

#### Slow Inference

```bash
# 1. Increase GPU layers
model.n_gpu_layers: 99

# 2. Reduce context window
model.n_ctx: 4096  # Instead of 8192

# 3. Use quantized model
# Q4_K_M instead of FP16
```

#### Slow RAG Queries

```bash
# 1. Reduce top_k
rag.top_k: 3  # Instead of 5

# 2. Use faster embedding model
rag.embedding_model: "all-MiniLM-L6-v2"  # Already fast

# 3. Pre-compute embeddings
# (Ingest once, query many times)
```

---

## Architecture

```
inference-server/
├── config.yaml              # Configuration
├── models/                  # Model files (GGUF/safetensors)
│   └── model.gguf
├── rag_data/
│   ├── documents/           # Source documents
│   │   ├── project1.pdf
│   │   ├── terrain.docx
│   │   └── specs.txt
│   └── store/               # ChromaDB vector store (auto-created)
├── src/
│   ├── server.py            # FastAPI server (OpenAI-compatible)
│   ├── inference.py         # Model inference engine
│   ├── rag.py               # RAG + document ingestion
│   ├── config.py            # Configuration management
│   └── cli.py               # CLI interface
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

### Request Flow

```
User Request
    │
    ▼
┌─────────────────┐
│  FastAPI Server  │
│  (server.py)    │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ Model  │ │  RAG   │
│ Engine │ │ Store  │
└────┬───┘ └───┬────┘
     │         │
     │    ┌────┴────┐
     │    ▼         │
     │ ┌────────┐   │
     │ │ChromaDB│   │
     │ └────┬───┘   │
     │      │       │
     │   ┌──┴───┐   │
     │   │Embed │   │
     │   │dings │   │
     │   └──────┘   │
     │              │
     └──────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  Response +   │
    │  Sources      │
    └───────────────┘
```

---

## Cost Analysis

### Self-Hosted (Recommended)

| Component | Cost (PLN) |
|-----------|------------|
| Server (RTX 3090) | 15,000-20,000 one-time |
| Electricity | 200-400/month |
| Internet | Existing |
| Maintenance | Your time |
| **Year 1 Total** | ~20,000-25,000 |
| **Year 2+ Total** | ~2,400-4,800/year |

### Cloud API Comparison

| Provider | Cost (1M tokens/day) |
|----------|---------------------|
| OpenAI GPT-4 | ~3,000-5,000/month |
| Claude | ~2,000-4,000/month |
| Local Llama 7B | ~200-400/month (electricity) |

### Break-Even

- **vs OpenAI GPT-4**: 6-12 months
- **vs Claude**: 8-14 months
- **After break-even**: 90%+ cost reduction

---

## Python Usage

### Direct Python Usage

```python
from src.config import load_config
from src.inference import InferenceEngine
from src.rag import RAGStore, DocumentIngestor, RAGQuery

# Load config
config = load_config("config.yaml")

# Load model
engine = InferenceEngine()
engine.load(config.model.path, config)

# Standard inference
result = engine.generate(
    [{"role": "user", "content": "Hello!"}],
    max_tokens=512,
    temperature=0.7,
)
print(result["response"])

# RAG query
store = RAGStore(config.rag.store_path)
rag = RAGQuery(store, config.rag)
result = rag.query(engine, "What projects?", max_tokens=512)
print(result["response"])
print(f"Sources: {result['sources']}")
```

### Batch Processing

```python
import json
from src.config import load_config
from src.inference import InferenceEngine

config = load_config()
engine = InferenceEngine()
engine.load(config.model.path, config)

# Process questions from file
with open("questions.jsonl") as f:
    for line in f:
        question = json.loads(line)["question"]
        result = engine.generate(
            [{"role": "user", "content": question}],
            max_tokens=512,
        )
        print(json.dumps({
            "question": question,
            "answer": result["response"],
            "time_ms": result["time_ms"],
        }))
```

---

## Support

- **Issues**: GitHub Issues
- **Documentation**: This file
- **Examples**: See `examples/` directory

---

**Version**: 1.0.0
**Last Updated**: 2026-08-17
