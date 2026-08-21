"""FastAPI server v1 — basic OpenAI-compatible chat endpoint.

WHAT THIS FILE DOES
==================
The first version of the HTTP API. Compatible with OpenAI's chat
completions format, so any OpenAI client library (Python, JS, curl)
can talk to it without modification.

KEY CONCEPTS
============
- FastAPI: a modern Python web framework. Defines API endpoints as
  Python functions with type annotations.
- OpenAI compatibility: the request/response format matches OpenAI's
  API, so existing clients work without changes.
- Async: FastAPI uses async/await for non-blocking I/O.
- Streaming: tokens can be sent one-by-one as the model generates them.
"""

"""Portable Inference Server with RAG — main server."""
from contextlib import asynccontextmanager
import os
from pathlib import Path
import time

import aiofiles
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import AppConfig, load_config
from .inference import InferenceEngine
from .parsers import ingest_bytes, parse_document, supported_extensions
from .rag import DocumentIngestor, RAGStore

# ── Globals ──
config: AppConfig = None
engine: InferenceEngine = None
rag_store: RAGStore = None
ingestor: DocumentIngestor = None
agent = None
start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, engine, rag_store, ingestor, agent, start_time

    start_time = time.time()
    config = load_config()

    # Load model
    engine = InferenceEngine()
    if config.model.path and os.path.exists(config.model.path):
        print(f"Loading model: {config.model.path}")
        engine.load(config.model.path, config)
        print("Model loaded!")
    else:
        print("No model configured or found")

    # Setup RAG
    if config.rag.enabled:
        rag_store = RAGStore(config.rag.store_path)
        ingestor = DocumentIngestor(rag_store, config.rag.chunk_size, config.rag.chunk_overlap)

        # Auto-ingest documents on startup
        if os.path.exists(config.rag.documents_path):
            print(f"Ingesting documents from: {config.rag.documents_path}")
            result = ingestor.ingest_directory(
                config.rag.documents_path, embedding_model=config.rag.embedding_model
            )
            print(f"Ingested {result['files_ingested']} files, {result['chunks_added']} chunks")
            if result["errors"]:
                print(f"Errors: {len(result['errors'])}")

        # Setup agent
        from .agent import ToolCallingAgent

        agent = ToolCallingAgent(engine, rag_store, config.rag.embedding_model)

    print(f"Server ready on http://{config.server.host}:{config.server.port}")
    yield

    # Cleanup
    if engine:
        engine.unload()


app = FastAPI(
    title="Inference Server",
    version="1.1.0",
    lifespan=lifespan,
)

if config and config.api.cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ── Auth ──
def verify_api_key(authorization: str | None = Header(None)):
    if config and config.api.key:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing API key")
        token = authorization.replace("Bearer ", "")
        if token != config.api.key:
            raise HTTPException(status_code=401, detail="Invalid API key")


# ── Models ──
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    agentic: bool = False  # Enable tool calling agent loop
    tools: list = None


class RAGQueryRequest(BaseModel):
    question: str
    model: str = "default"
    max_tokens: int = 1024
    temperature: float = 0.7
    top_k: int = 5
    system_prompt: str = ""


class IngestRequest(BaseModel):
    path: str
    extensions: list = None


# ── OpenAI-Compatible Endpoints ──
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, _=Depends(verify_api_key)):  # noqa: B008
    if not engine or engine.model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Agentic mode — let the model call RAG tools
    if request.agentic:
        if agent is None:
            raise HTTPException(status_code=503, detail="Agent not available (RAG disabled)")
        result = agent.run(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["response"]},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "x_inference_time_ms": 0,
            "x_tool_calls": result["tool_calls"],
        }

    # Standard mode
    result = engine.generate(
        messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["response"]},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "x_inference_time_ms": result["time_ms"],
    }


@app.get("/v1/models")
async def list_models(_=Depends(verify_api_key)):  # noqa: B008
    models = []
    if engine and engine.model_path:
        models.append(
            {
                "id": "default",
                "object": "model",
                "created": int(start_time),
                "owned_by": "local",
            }
        )
    return {"object": "list", "data": models}


# ── RAG Endpoints ──
@app.post("/v1/rag/query")
async def rag_query(request: RAGQueryRequest, _=Depends(verify_api_key)):  # noqa: B008
    if not engine or engine.model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    if rag_store is None:
        raise HTTPException(status_code=503, detail="RAG not enabled")

    # Retrieve context
    results = rag_store.search(
        request.question, top_k=request.top_k, embedding_model=config.rag.embedding_model
    )

    # Build context
    context_parts = []
    for r in results:
        if r.score >= config.rag.min_score:
            context_parts.append(f"[Source: {r.source}]\n{r.text}")

    context = "\n\n---\n\n".join(context_parts[: config.rag.top_k])

    # Build messages
    messages = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})

    if context:
        rag_prompt = f"Based on the following documents:\n\n{context}\n\nAnswer the question."
        messages.append({"role": "system", "content": rag_prompt})

    messages.append({"role": "user", "content": request.question})

    # Generate
    result = engine.generate(
        messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return {
        "response": result["response"],
        "sources": [
            {"text": r.text[:300], "score": round(r.score, 3), "source": r.source}
            for r in results
            if r.score >= config.rag.min_score
        ],
        "chunks_retrieved": len([r for r in results if r.score >= config.rag.min_score]),
        "time_ms": result["time_ms"],
    }


@app.post("/v1/rag/ingest")
async def rag_ingest(file: UploadFile = File(...), _=Depends(verify_api_key)):  # noqa: B008
    if rag_store is None:
        raise HTTPException(status_code=503, detail="RAG not enabled")

    content = await file.read()

    # Parse and ingest directly from bytes (any office format)
    result = ingest_bytes(file.filename, content, rag_store, config.rag.embedding_model)

    # Also save to documents dir for provenance
    upload_dir = Path(config.rag.documents_path)
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    async with aiofiles.open(file_path, "wb") as f:
        f.write(content)

    result["saved_to"] = str(file_path)
    return result


@app.post("/v1/rag/ingest-directory")
async def rag_ingest_directory(request: IngestRequest, _=Depends(verify_api_key)):  # noqa: B008
    if rag_store is None:
        raise HTTPException(status_code=503, detail="RAG not enabled")

    result = ingestor.ingest_directory(
        request.path, request.extensions, embedding_model=config.rag.embedding_model
    )
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


# ── Document Parsing Endpoints ──
@app.get("/v1/parse/supported")
async def parse_supported():
    """List supported document formats."""
    return {"formats": supported_extensions()}


@app.post("/v1/parse")
async def parse_file(file: UploadFile = File(...), _=Depends(verify_api_key)):  # noqa: B008
    """Parse a document and return its text content."""
    content = await file.read()
    text = parse_document(file.filename)  # Fallback for server-side files
    # For uploads, use bytes parser
    from .parsers import parse_bytes

    text = parse_bytes(file.filename, content)
    return {
        "filename": file.filename,
        "text": text[:20000],
        "length": len(text),
    }


# ── Management ──
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": engine.model is not None if engine else False,
        "rag_enabled": rag_store is not None,
        "agent_enabled": agent is not None,
        "uptime_seconds": round(time.time() - start_time, 1),
    }


@app.get("/stats")
async def stats(_=Depends(verify_api_key)):  # noqa: B008
    return {
        "model": engine.status() if engine else {},
        "rag": {
            "enabled": rag_store is not None,
            "total_chunks": rag_store.count() if rag_store else 0,
            "documents": len(rag_store.list_documents()) if rag_store else 0,
        },
        "supported_document_formats": len(supported_extensions()),
        "uptime_seconds": round(time.time() - start_time, 1),
    }


@app.post("/reload")
async def reload_model(_=Depends(verify_api_key)):  # noqa: B008
    """Reload model and RAG from config."""
    global engine, rag_store, ingestor, agent

    if engine:
        engine.unload()

    engine = InferenceEngine()
    if config.model.path and os.path.exists(config.model.path):
        engine.load(config.model.path, config)

    if config.rag.enabled:
        rag_store = RAGStore(config.rag.store_path)
        ingestor = DocumentIngestor(rag_store, config.rag.chunk_size, config.rag.chunk_overlap)
        from .agent import ToolCallingAgent

        agent = ToolCallingAgent(engine, rag_store, config.rag.embedding_model)

    return {"status": "reloaded", "model": engine.status()}
