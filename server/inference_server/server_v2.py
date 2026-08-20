"""FastAPI server v2 — adds RAG, tool calling, MCP, file upload.

WHAT THIS FILE DOES
==================
The production version of the HTTP API. Extends server.py with:
  - File upload for RAG (POST /v1/rag/ingest with multipart/form-data)
  - Tool calling endpoints
  - MCP server integration
  - Agentic loop execution

KEY CONCEPTS
============
- Multipart form data: a way to upload files in HTTP requests.
- Model Context Protocol (MCP): a standard for giving LLMs access to tools.
- Dependency injection: FastAPI's way of providing shared resources
  (like the RAG store) to endpoint functions.
"""

"""Portable Inference Server v2 — with tool calling, samplers, MCP, and RAG."""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import aiofiles
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import load_config
from .inference import InferenceEngine
from .rag import DocumentIngestor, RAGStore
from .samplers import PRESETS, SamplerConfig

# ── Globals ──
config = None
engine = None
rag_store = None
ingestor = None
mcp_server = None
start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, engine, rag_store, ingestor, mcp_server, start_time

    start_time = time.time()
    config = load_config()

    # Load model
    engine = InferenceEngine()
    if config.model.path and os.path.exists(config.model.path):
        print(f"Loading model: {config.model.path}")
        engine.load(config.model.path, config)
        print("Model loaded!")

    # Setup RAG
    if config.rag.enabled:
        rag_store = RAGStore(config.rag.store_path)
        ingestor = DocumentIngestor(rag_store, config.rag.chunk_size, config.rag.chunk_overlap)

        # Auto-ingest documents
        if os.path.exists(config.rag.documents_path):
            print(f"Ingesting documents from: {config.rag.documents_path}")
            result = ingestor.ingest_directory(config.rag.documents_path, embedding_model=config.rag.embedding_model)
            print(f"Ingested {result['files_ingested']} files, {result['chunks_added']} chunks")

        # Setup MCP server
        from .mcp import RAGMCPServer
        mcp_server = RAGMCPServer(rag_store, config.rag.embedding_model)

    print(f"Server ready on http://{config.server.host}:{config.server.port}")
    yield

    if engine:
        engine.unload()


app = FastAPI(title="Inference Server", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Models ──
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    min_p: float = 0.05
    stream: bool = False
    sampler_preset: str | None = None
    agentic: bool = False
    tools: list = None

class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 5
    max_tokens: int = 512
    temperature: float = 0.7
    system_prompt: str = ""


# ── Auth ──
def verify_api_key(authorization: str | None = Header(None)):
    if config and config.api.key:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing API key")
        token = authorization.replace("Bearer ", "")
        if token != config.api.key:
            raise HTTPException(status_code=401, detail="Invalid API key")


# ── Endpoints ──
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, _=Depends(verify_api_key)):  # noqa: B008
    if not engine or engine.model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Apply sampler preset
    sampler = SamplerConfig(
        temperature=request.temperature, top_p=request.top_p, top_k=request.top_k,
        repeat_penalty=request.repeat_penalty, min_p=request.min_p, max_tokens=request.max_tokens,
    )
    if request.sampler_preset and request.sampler_preset in PRESETS:
        preset = PRESETS[request.sampler_preset]
        sampler.temperature = preset.temperature
        sampler.top_p = preset.top_p
        sampler.top_k = preset.top_k
        sampler.repeat_penalty = preset.repeat_penalty
        sampler.min_p = preset.min_p

    # Agentic mode with tool calling
    if request.agentic and mcp_server:
        from ..agent import ToolCallingAgent
        agent = ToolCallingAgent(engine, rag_store, config.rag.embedding_model if rag_store else "all-MiniLM-L6-v2")
        result = agent.run(messages, max_tokens=sampler.max_tokens, temperature=sampler.temperature)
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result["response"]}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "x_tool_calls": result.get("tool_calls", []),
            "x_sampler": {"temperature": sampler.temperature, "top_p": sampler.top_p, "top_k": sampler.top_k},
        }

    # Standard generation
    result = engine.generate(messages, max_tokens=sampler.max_tokens, temperature=sampler.temperature, top_p=sampler.top_p)
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": result if isinstance(result, str) else result.get("response", str(result))}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "x_sampler": {"temperature": sampler.temperature, "top_p": sampler.top_p, "top_k": sampler.top_k},
    }


@app.get("/v1/models")
async def list_models(_=Depends(verify_api_key)):  # noqa: B008
    models = []
    if engine and engine.model_path:
        models.append({"id": "default", "object": "model", "created": int(start_time), "owned_by": "local"})
    return {"object": "list", "data": models}


@app.get("/v1/samplers")
async def list_samplers():
    """List available sampler presets."""
    from .samplers import list_presets
    return {
        "presets": list_presets(),
        "defaults": {
            "temperature": 0.7, "top_p": 0.9, "top_k": 40,
            "repeat_penalty": 1.1, "min_p": 0.05,
        }
    }


# ── RAG Endpoints ──
@app.post("/v1/rag/query")
async def rag_query(request: RAGQueryRequest, _=Depends(verify_api_key)):  # noqa: B008
    if not engine or engine.model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    if rag_store is None:
        raise HTTPException(status_code=503, detail="RAG not enabled")

    results = rag_store.search(request.question, top_k=request.top_k, embedding_model=config.rag.embedding_model)
    context_parts = [f"[Source: {r.source}]\n{r.text}" for r in results if r.score >= config.rag.min_score]
    context = "\n\n---\n\n".join(context_parts)

    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    if context:
        messages.append({"role": "system", "content": f"Based on the following documents:\n\n{context}\n\nAnswer the question using this information when relevant."})
    messages.append({"role": "user", "content": request.question})

    result = engine.generate(messages, max_tokens=request.max_tokens, temperature=request.temperature)
    return {
        "response": result if isinstance(result, str) else result.get("response", str(result)),
        "sources": [{"text": r.text[:200], "score": round(r.score, 3), "source": r.source} for r in results if r.score >= config.rag.min_score],
        "chunks_retrieved": len([r for r in results if r.score >= config.rag.min_score]),
    }


@app.post("/v1/rag/ingest")
async def rag_ingest(file: UploadFile = File(...), _=Depends(verify_api_key)):  # noqa: B008
    if rag_store is None:
        raise HTTPException(status_code=503, detail="RAG not enabled")
    content = await file.read()
    from .parsers import ingest_bytes
    result = ingest_bytes(file.filename, content, rag_store, config.rag.embedding_model)
    # Save file
    upload_dir = Path(config.rag.documents_path)
    upload_dir.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(upload_dir / file.filename, "wb") as f:
        f.write(content)
    result["saved_to"] = str(upload_dir / file.filename)
    return result


@app.post("/v1/rag/ingest-directory")
async def rag_ingest_directory(request: Request, _=Depends(verify_api_key)):  # noqa: B008
    if rag_store is None:
        raise HTTPException(status_code=503, detail="RAG not enabled")
    body = await request.json()
    path = body.get("path", "")
    result = ingestor.ingest_directory(path, embedding_model=config.rag.embedding_model)
    return result


@app.get("/v1/rag/documents")
async def rag_list_documents(_=Depends(verify_api_key)):  # noqa: B008
    if rag_store is None:
        raise HTTPException(status_code=503, detail="RAG not enabled")
    return {"documents": rag_store.list_documents(), "total_chunks": rag_store.count()}


@app.delete("/v1/rag/documents/{document_id}")
async def rag_remove_document(document_id: str, _=Depends(verify_api_key)):  # noqa: B008
    if rag_store is None:
        raise HTTPException(status_code=503, detail="RAG not enabled")
    removed = rag_store.remove_document(document_id)
    return {"document_id": document_id, "chunks_removed": removed}


# ── MCP Endpoints ──
@app.get("/v1/mcp/tools")
async def mcp_list_tools():
    """List available MCP tools."""
    if mcp_server is None:
        return {"tools": [], "error": "MCP server not initialized"}
    return {"tools": mcp_server.list_tools(), "server": mcp_server.to_dict()}


@app.post("/v1/mcp/execute")
async def mcp_execute_tool(request: Request):
    """Execute an MCP tool."""
    if mcp_server is None:
        raise HTTPException(status_code=503, detail="MCP server not initialized")
    body = await request.json()
    tool_name = body.get("tool", "")
    arguments = body.get("arguments", {})
    result = mcp_server.execute_tool(tool_name, arguments)
    return json.loads(result) if isinstance(result, str) else result


# ── Management ──
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": engine.model is not None if engine else False,
        "rag_enabled": rag_store is not None,
        "mcp_enabled": mcp_server is not None,
        "uptime_seconds": round(time.time() - start_time, 1),
    }


@app.get("/stats")
async def stats(_=Depends(verify_api_key)):  # noqa: B008
    return {
        "model": engine.status() if engine else {},
        "rag": {"enabled": rag_store is not None, "total_chunks": rag_store.count() if rag_store else 0, "documents": len(rag_store.list_documents()) if rag_store else 0},
        "mcp": {"enabled": mcp_server is not None, "tools": len(mcp_server.tools) if mcp_server else 0},
        "samplers": list(PRESETS.keys()),
        "uptime_seconds": round(time.time() - start_time, 1),
    }


@app.post("/reload")
async def reload_model(_=Depends(verify_api_key)):  # noqa: B008
    global engine, rag_store, ingestor, mcp_server
    if engine:
        engine.unload()
    engine = InferenceEngine()
    if config.model.path and os.path.exists(config.model.path):
        engine.load(config.model.path, config)
    if config.rag.enabled:
        rag_store = RAGStore(config.rag.store_path)
        ingestor = DocumentIngestor(rag_store, config.rag.chunk_size, config.rag.chunk_overlap)
        from .mcp import RAGMCPServer
        mcp_server = RAGMCPServer(rag_store, config.rag.embedding_model)
    return {"status": "reloaded", "model": engine.status() if engine else {}}
