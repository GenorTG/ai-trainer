# ADR-001: Why llama.cpp over vLLM for inference

**Status**: Accepted
**Date**: 2026-08-20
**Decision**: Use llama.cpp + llama-cpp-python for inference, NOT vLLM or TGI.

## Context

We need a fast, lightweight inference server that:
- Runs on consumer hardware (RTX 3090 24GB, GTX 1070 8GB)
- Supports GGUF quantized models (Q4_K_M, Q5_K_M)
- Has OpenAI-compatible API
- Works with our small 14B-parameter Gemma 4 model
- Doesn't require 100GB+ system RAM
- Is simple to deploy on a home server

## Options Considered

| Option | Pros | Cons |
|--------|------|------|
| **llama.cpp** ✅ | Pure C++, minimal deps, supports GGUF, runs on any GPU/CPU | Slower than vLLM for huge models |
| vLLM | High throughput, PagedAttention | Needs CUDA 12+, complex setup, doesn't support GGUF |
| TGI (HF) | Production-ready, batching | Heavy, needs Docker, no GGUF |
| Ollama | Simple, popular | Not OpenAI-compatible, hides config |
| LM Studio | GUI-only | Not scriptable |

## Decision

Use **llama.cpp** + **llama-cpp-python** because:

1. **GGUF support** — Our v20/v21 models are Q4_K_M GGUF. vLLM/TGI require HF format conversion.
2. **Low memory** — Runs on GTX 1070 8GB with `-ngl 99` partial offload
3. **Simple deployment** — Single binary + Python wrapper, no Docker required
4. **Pure inference** — No training, no embedding server, no token counting fluff
5. **Familiar** — Already deployed on genorbox1 for production

## Consequences

- **Pro**: Easy to deploy, low memory footprint, fast startup
- **Pro**: Works on both RTX 3090 (fan-dragon) and GTX 1070 (genorbox1)
- **Con**: Slower than vLLM for batch processing (we don't batch)
- **Con**: No built-in observability — we wrap with custom metrics
- **Con**: Limited tool-calling — we wrote custom parser for Gemma4 format

## Implementation

```bash
# llama.cpp + llama-cpp-python
pip install llama-cpp-python

# Serve a model
inference-server serve --model models/v21/chris-ai-gemma4-e4b-v21.Q4_K_M.gguf

# GPU offload
inference-server serve --model ... --n-gpu-layers 99
```

## Future

If we ever need >1000 tokens/sec throughput or 70B+ models, switch to vLLM. For our 14B model with ~50 tok/s on RTX 3090, llama.cpp is sufficient.