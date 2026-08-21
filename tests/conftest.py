"""Shared test fixtures and mocks for ALL tests (DRY principle).

WHAT THIS FILE DOES
===================
Single source of truth for test fixtures. Every test file imports from here.
Avoids duplicating mock setup across test files.

KEY CONCEPTS
============
- pytest fixtures: dependencies tests need, built once and reused
- mocks via pytest-mock (`mocker` fixture): drop-in replacement for objects
- tmp_path: pytest's built-in temp directory fixture
- DRY: this file is the canonical place for shared test infrastructure

AVAILABLE FIXTURES
==================
HTTP/API:
  - client (FastAPI TestClient)
  - async_client (httpx async)
  - inference_server_client (inference-server TestClient)
  - finetune_studio_client (finetune-studio WebUI TestClient)

Models:
  - mock_engine (mock llama.cpp InferenceEngine)
  - mock_model_path (path to a fake GGUF file)
  - mock_embeddings (mock sentence transformers)

RAG:
  - mock_rag_store (mock ChromaDB)
  - mock_rag_documents (sample documents)

Tools:
  - sample_messages
  - sample_tools
  - sample_tool_call

Files:
  - tmp_dir (writable temp directory)
  - sample_jsonl (training data file)
  - sample_text_file
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure both packages are importable
ROOT = Path(__file__).resolve().parent.parent

# Support both ai-trainer and finetune-studio layouts
_TRAINER_SRC = ROOT / "trainer" / "src"
_SERVER_SRC = ROOT / "server"
_FINETUNE_SRC = ROOT / "finetune-studio" / "src"
_INFERENCE_SRC = ROOT / "inference-server"

for p in [_TRAINER_SRC, _SERVER_SRC, _FINETUNE_SRC, _INFERENCE_SRC]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ============================================================================
# FILE SYSTEM FIXTURES
# ============================================================================

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Writable temp directory that auto-cleans."""
    d = tmp_path / "test_workspace"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def sample_text_file(tmp_dir: Path) -> Path:
    """A simple .txt file for parsing tests."""
    p = tmp_dir / "sample.txt"
    p.write_text("Hello world. This is a test document about AI and ML.")
    return p


@pytest.fixture
def sample_jsonl(tmp_dir: Path) -> Path:
    """A training-data JSONL file."""
    p = tmp_dir / "training.jsonl"
    samples = [
        {"messages": [{"role": "user", "content": "What is AI?"},
                       {"role": "assistant", "content": "Artificial Intelligence."}]},
        {"messages": [{"role": "user", "content": "What is ML?"},
                       {"role": "assistant", "content": "Machine Learning."}]},
    ]
    with p.open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return p


@pytest.fixture
def sample_pdf(tmp_dir: Path) -> Path:
    """Minimal PDF-like bytes (used for parser structural tests)."""
    p = tmp_dir / "sample.pdf"
    p.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF")
    return p


@pytest.fixture
def mock_model_path(tmp_dir: Path) -> Path:
    """Fake GGUF model file with minimal GGUF magic bytes."""
    p = tmp_dir / "model.gguf"
    # GGUF magic: 'GGUF' + version + counts + metadata (we use empty/zero for mocks)
    p.write_bytes(b"GGUF" + b"\x03\x00\x00\x00" + b"\x00" * 1024)
    return p


# ============================================================================
# MODEL / INFERENCE MOCKS
# ============================================================================

@pytest.fixture
def mock_engine(mocker):
    """Mock InferenceEngine — returns deterministic text without loading GGUF."""
    engine = MagicMock()
    engine.is_loaded = True
    engine.model_path = "/fake/model.gguf"
    engine.generate = MagicMock(return_value={
        "response": "Mock response from Chris AI.",
        "tokens": 42,
        "time_ms": 100.5,
    })
    engine.unload = MagicMock()
    engine.load = MagicMock()
    return engine


@pytest.fixture
def mock_llama(mocker):
    """Mock the llama_cpp library (skips actual GGUF loading).

    Used to test imports / instantiation without GPU/CPU work.
    """
    fake_llama = MagicMock()
    fake_llama.Llama = MagicMock(return_value=MagicMock(
        __call__=MagicMock(return_value={
            "choices": [{"text": "Mock response"}],
            "usage": {"total_tokens": 10},
        }),
    ))
    mocker.patch.dict(sys.modules, {"llama_cpp": fake_llama})
    return fake_llama


# ============================================================================
# RAG MOCKS
# ============================================================================

@pytest.fixture
def mock_rag_documents() -> list[dict]:
    """Sample RAG documents for testing ingest/search."""
    return [
        {
            "id": "doc1",
            "text": "RAG (Retrieval-Augmented Generation) is a technique where a model retrieves relevant documents before generating an answer.",
            "source": "rag_intro.md",
            "metadata": {"category": "ml"},
        },
        {
            "id": "doc2",
            "text": "Fine-tuning adapts a pre-trained LLM to a specific domain by further training on domain data.",
            "source": "finetune.md",
            "metadata": {"category": "ml"},
        },
        {
            "id": "doc3",
            "text": "GGUF is a file format for distributing large language models for inference.",
            "source": "gguf.md",
            "metadata": {"category": "format"},
        },
    ]


@pytest.fixture
def mock_rag_store(mocker, mock_rag_documents):
    """Mock ChromaDB-backed RAG store that returns canned results."""
    store = MagicMock()
    store.search = MagicMock(return_value=[
        MagicMock(text=d["text"], score=0.9 - i * 0.1, source=d["source"], metadata=d["metadata"])
        for i, d in enumerate(mock_rag_documents)
    ])
    store.add = MagicMock()
    store.list_documents = MagicMock(return_value=[d["id"] for d in mock_rag_documents])
    store.delete = MagicMock()
    store.ingest_file = MagicMock(return_value=3)
    store.ingest_directory = MagicMock(return_value=10)
    return store


@pytest.fixture
def mock_embeddings(mocker):
    """Mock embedding model — returns deterministic unit vectors."""
    fake_model = MagicMock()
    fake_model.encode = MagicMock(return_value=MagicMock(
        tolist=lambda: [[0.1] * 384]
    ))
    mocker.patch(
        "sentence_transformers.SentenceTransformer",
        return_value=fake_model,
        create=True,
    )
    return fake_model


# ============================================================================
# CHAT / TOOL MOCKS
# ============================================================================

@pytest.fixture
def sample_messages() -> list[dict]:
    """Standard chat messages for testing."""
    return [
        {"role": "system", "content": "You are Chris AI, a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"},
    ]


@pytest.fixture
def sample_tools() -> list[dict]:
    """Sample tool definitions (OpenAI function-calling format)."""
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Calculate a math expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"],
                },
            },
        },
    ]


@pytest.fixture
def sample_tool_call() -> dict:
    """A parsed tool call result for testing."""
    return {
        "name": "web_search",
        "arguments": {"query": "weather in Warsaw"},
    }


@pytest.fixture
def mock_tool_response() -> str:
    """A model response containing a tool call."""
    return (
        '<tool_call>\n'
        '{"name": "web_search", "arguments": {"query": "Warsaw weather"}}\n'
        '</tool_call>'
    )


# ============================================================================
# FASTAPI / HTTP CLIENT FIXTURES
# ============================================================================

@pytest.fixture
def fastapi_test_client():
    """Generic FastAPI TestClient — use when you import a specific app."""
    from fastapi.testclient import TestClient
    return TestClient


@pytest.fixture
def inference_server_client(mocker, mock_engine, mock_rag_store):
    """TestClient for the inference-server /v1 API.

    Mocks out heavy resources: GGUF loader, RAG store, MCP server.
    Returns a FastAPI TestClient ready to make requests.
    """
    # Patch BEFORE importing the server module
    mocker.patch("inference_server.inference.InferenceEngine", return_value=mock_engine)
    mocker.patch("inference_server.rag.RAGStore", return_value=mock_rag_store)

    from fastapi.testclient import TestClient
    from inference_server.server import app

    return TestClient(app)


@pytest.fixture
def inference_server_v2_client(mocker, mock_engine, mock_rag_store):
    """TestClient for the inference-server v2 API (with MCP, samplers)."""
    mocker.patch("inference_server.inference.InferenceEngine", return_value=mock_engine)
    mocker.patch("inference_server.rag.RAGStore", return_value=mock_rag_store)

    from fastapi.testclient import TestClient
    from inference_server.server_v2 import app

    return TestClient(app)


@pytest.fixture
def finetune_studio_client(mocker, mock_engine, mock_rag_store):
    """TestClient for finetune-studio WebUI on port 7860."""
    # IMPORTANT: import the module BEFORE mocker.patch can resolve the path.
    # mocker.patch uses pkgutil.resolve_name which requires the package to
    # already have the attribute. Importing here forces the module to load.
    import finetune_studio.webui.app as _fs_app  # noqa: F401

    mocker.patch("finetune_studio.webui.app.scan_models", return_value=["/fake/model.gguf"])
    mocker.patch("finetune_studio.webui.app.discovered_models", ["fake_model"], create=True)
    mocker.patch("finetune_studio.webui.app.inference_engine", mock_engine, create=True)
    mocker.patch("finetune_studio.webui.app.training_engine", MagicMock(), create=True)

    from fastapi.testclient import TestClient
    from finetune_studio.webui.app import app

    return TestClient(app)


# ============================================================================
# MOCK GPU / CUDA
# ============================================================================

@pytest.fixture
def mock_cuda(mocker):
    """Mock torch.cuda so tests don't require GPU."""
    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.cuda.device_count", return_value=0)


@pytest.fixture
def mock_no_comfyui(mocker):
    """Mock ComfyUI detection so tests don't touch real GPU services."""
    mocker.patch("subprocess.run", return_value=MagicMock(returncode=1, stdout=b""))


# ============================================================================
# ASYNC HELPERS
# ============================================================================

@pytest.fixture
def event_loop():
    """Single asyncio event loop per test (avoids 'loop closed' warnings)."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# CLEANUP HELPERS
# ============================================================================

@pytest.fixture(autouse=True)
def reset_env():
    """Snapshot env vars, restore after test."""
    saved = os.environ.copy()
    yield
    # Restore only known keys
    for key in list(os.environ.keys()):
        if key not in saved:
            os.environ.pop(key, None)
    for key, val in saved.items():
        os.environ[key] = val


@pytest.fixture(autouse=True)
def cleanup_caches():
    """Clear module-level caches between tests."""
    yield
    # Force reimport of changed modules
    import importlib
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith(("finetune_studio", "inference_server")):
            importlib.reload(sys.modules[mod_name])


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Register custom markers used across the test suite."""
    config.addinivalue_line("markers", "unit: unit test (fast, no I/O)")
    config.addinivalue_line("markers", "integration: integration test (multiple components)")
    config.addinivalue_line("markers", "api: HTTP/API endpoint test")
    config.addinivalue_line("markers", "frontend: HTML/JS/browser test")
    config.addinivalue_line("markers", "slow: slow test (model loading, GPU)")
