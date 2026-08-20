# Architecture Overview

High-level view of AI Trainer system components and data flow.

## System Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          USER (genorbox1 workspace)                          │
│                                                                              │
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────────┐               │
│  │  Webchat     │───►│  Discord Bot  │───►│  Vercel Web     │               │
│  │  (Amy/Claude)│    │  (commands)   │    │  ai.smart-      │               │
│  └──────────────┘    └───────────────┘    │  samurai.pl     │               │
│                                            └────────┬────────┘               │
└─────────────────────────────────────────────────────┼──────────────────────┘
                                                      │
                                                      │ HTTPS + Bearer token
                                                      │
                       ┌──────────────────────────────▼──────────────────────┐
                       │        PRODUCTION (genorbox1)                       │
                       │   ┌────────────────────────────────────────────┐     │
                       │   │  Nginx reverse proxy                       │     │
                       │   │  → chris-ai.service (systemd)              │     │
                       │   │  → serve.py (HTTP wrapper)                 │     │
                       │   │  → llama-server (llama.cpp, port 8088)    │     │
                       │   │  → v20 GGUF (Q4_K_M, 5.0GB)               │     │
                       │   └────────────────────────────────────────────┘     │
                       └──────────────────────────────┬──────────────────────┘
                                                      │
                                                      │ Tailscale mesh
                                                      │
                       ┌──────────────────────────────▼──────────────────────┐
                       │      DEVELOPMENT (fan-dragon)                       │
                       │                                                      │
                       │   ┌──────────────────────────────────────────────┐   │
                       │   │            ~/ai-trainer/                     │   │
                       │   │                                              │   │
                       │   │   ┌──────────────────────────────────────┐   │   │
                       │   │   │  trainer/ (finetune_studio pkg)     │   │   │
                       │   │   │  - WebUI (FastAPI, port 7860)       │   │   │
                       │   │   │  - CLI (fts)                        │   │   │
                       │   │   │  - Training (QLoRA, unsloth)       │   │   │
                       │   │   │  - Benchmarks (MMLU, HellaSwag)    │   │   │
                       │   │   │  - RAG + Tool calling               │   │   │
                       │   │   └──────────────────────────────────────┘   │   │
                       │   │                                              │   │
                       │   │   ┌──────────────────────────────────────┐   │   │
                       │   │   │  server/ (inference_server pkg)     │   │   │
                       │   │   │  - v1 OpenAI-compatible (port 8888) │   │   │
                       │   │   │  - v2 with RAG + tool calling             │   │   │
                       │   │   │  - 33 doc parsers                   │   │   │
                       │   │   │  - MCP tools (search, calc, etc)  │   │   │
                       │   │   │  - Jinja template extraction       │   │   │
                       │   │   └──────────────────────────────────────┘   │   │
                       │   │                                              │   │
                       │   │   ┌──────────────────────────────────────┐   │   │
                       │   │   │  data/                                │   │   │
                       │   │   │  - data_v21_training.jsonl            │   │   │
                       │   │   │  - data_v21_knowledge.jsonl (3% mix)  │   │   │
                       │   │   │  - gen_*.py (synthetic gen scripts)   │   │   │
                       │   │   └──────────────────────────────────────┘   │   │
                       │   │                                              │   │
                       │   │   ┌──────────────────────────────────────┐   │   │
                       │   │   │  tests/                               │   │   │
                       │   │   │  - 573 tests, 64% coverage            │   │   │
                       │   │   │  - unit/ api/ frontend/ subdirs       │   │   │
                       │   │   └──────────────────────────────────────┘   │   │
                       │   │                                              │   │
                       │   │   ┌──────────────────────────────────────┐   │   │
                       │   │   │  docs/                                │   │   │
                       │   │   │  - README.md (index)                  │   │   │
                       │   │   │  - adr/ (architecture decisions)      │   │   │
                       │   │   │  - per-module READMEs                 │   │   │
                       │   │   └──────────────────────────────────────┘   │   │
                       │   │                                              │   │   │
                       │   │   ┌──────────────────────────────────────┐   │   │
                       │   │   │  tools/                               │   │   │
                       │   │   │  - deploy.sh (rsync to genorbox1)     │   │   │
                       │   │   │  - bump.py (semver bump)               │   │   │
                       │   │   │  - check-secrets.sh (pre-commit)      │   │   │
                       │   │   └──────────────────────────────────────┘   │   │
                       │   └──────────────────────────────────────────────┘   │
                       │                                                      │
                       │   ┌──────────────────────────────────────────────┐   │
                       │   │  ComfyUI (running, 10-17GB GPU)             │   │
                       │   │  GPU: RTX 3090 24GB                        │   │
                       │   │  → NEVER KILL, pause work when busy        │   │
                       │   └──────────────────────────────────────────────┘   │
                       └──────────────────────────────────────────────────────┘
```

## Data Flow: User Query → Response

```
1. User sends message via Discord/Webchat
        │
        ▼
2. Vercel website (Next.js) proxies to genorbox1:8888
        │
        ▼
3. chris-ai.service → llama-server (port 8088)
        │
        ▼
4. llama.cpp loads model, applies template (Jinja from GGUF)
        │
        ▼
5. Model generates tokens (v20 GGUF, Q4_K_M, 5.0GB)
        │
        ▼
6. llama-server streams back to Next.js
        │
        ▼
7. Next.js returns OpenAI-compatible response to user
```

## Data Flow: Training (fan-dragon only)

```
1. User runs `python trainer/train_v21.py` (or `fts train`)
        │
        ▼
2. Loads base Gemma 4 E4B model (HuggingFace format, 28GB)
        │
        ▼
3. Applies QLoRA (4-bit quantization + LoRA adapters)
        │
        ▼
4. Loads data_v21_training.jsonl (97% persona)
        │
        ▼
5. Loads data_v21_knowledge.jsonl (3% general knowledge)
        │
        ▼
6. Trains with TRL SFTTrainer + Unsloth optimization
        │
        ▼
7. Merges LoRA adapters into base model
        │
        ▼
8. Exports to GGUF (llama.cpp conversion script)
        │
        ▼
9. Quantizes to Q4_K_M (5.0GB)
        │
        ▼
10. Saves to models/v21/chris-ai-gemma4-e4b-v21.Q4_K_M.gguf
        │
        ▼
11. Deploys to genorbox1 production via tools/deploy.sh
```

## Data Flow: RAG Query

```
1. User asks question via /v1/chat/completions with `agentic: true`
        │
        ▼
2. server_v2.py dispatches to AgenticLoop
        │
        ▼
3. Model decides to call tool (e.g., rag_search)
        │
        ▼
4. RAGStore.search() with query embedding
        │
        ▼
5. ChromaDB returns top-k similar chunks
        │
        ▼
6. Chunks appended to messages as tool result
        │
        ▼
7. Model generates answer using retrieved context
        │
        ▼
8. Response streamed to client
```

## Deployment Topology

```
genorbox1 ─────────────────────────────────────────────────────► fan-dragon
   │                                                              │
   │  Tailscale                                                   │
   │ 100.125.137.97 (genorbox1)                            100.125.137.96 (fan-dragon)
   │                                                              │
   │ GTX 1070 8GB                                       RTX 3090 24GB
   │ production llama-server                              dev/benchmarks
   │ v20 GGUF (5.0GB)                                    v21 GGUF (5.3GB)
   │                                                          │
   │                                                          │ tools/deploy.sh
   │  ◄──────── rsync inference_server/ ─────────────────────  │
   │  ◄──────── rsync finetune_studio/  ─────────────────────  │
   │  ◄──────── rsync tests/             ─────────────────────  │
   │                                                          │
```

## See Also

- [docs/adr/001-llama-cpp.md](docs/adr/001-llama-cpp.md) — Why llama.cpp over vLLM
- [docs/adr/002-gemma-4.md](docs/adr/002-gemma-4.md) — Why Gemma 4 E4B over Llama 3
- [docs/adr/005-fan-dragon-primary.md](docs/adr/005-fan-dragon-primary.md) — Why fan-dragon as primary dev host