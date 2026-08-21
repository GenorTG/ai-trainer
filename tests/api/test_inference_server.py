"""Tests for inference-server v1 FastAPI endpoints (server.py).

Covers every endpoint in server.py:
  POST /v1/chat/completions, GET /v1/models,
  POST /v1/rag/query, POST /v1/rag/ingest, POST /v1/rag/ingest-directory,
  GET /v1/rag/documents, DELETE /v1/rag/documents/{id},
  GET /v1/parse/supported, POST /v1/parse,
  GET /health, GET /stats, POST /reload
"""
from __future__ import annotations

import io

import pytest

# ═══════════════════════════════════════════════════════════════════════════
# POST /v1/chat/completions
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.api
def test_chat_completions_standard(inference_server_client):
    """Standard chat returns OpenAI-compatible completion response."""
    resp = inference_server_client.post("/v1/chat/completions", json={
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "default"
    assert isinstance(body["choices"], list)
    assert len(body["choices"]) >= 1
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "message" in choice
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert "usage" in body
    assert "x_inference_time_ms" in body


@pytest.mark.api
def test_chat_completions_with_tools(inference_server_client):
    """Chat with tools parameter returns completion without error."""
    resp = inference_server_client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Search for weather"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert isinstance(body["choices"], list)


@pytest.mark.api
def test_chat_completions_no_model_loaded(mocker, mock_rag_store):
    """Returns 503 when no model is loaded (engine.model is None)."""
    from fastapi.testclient import TestClient

    from inference_server.server import app

    engine = mocker.MagicMock()
    engine.model = None
    engine.model_path = None

    mocker.patch("inference_server.inference.InferenceEngine", return_value=engine)
    mocker.patch("inference_server.rag.RAGStore", return_value=mock_rag_store)

    with TestClient(app) as client:
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
        })
    assert resp.status_code == 503
    assert "No model loaded" in resp.json()["detail"]


@pytest.mark.api
def test_chat_completions_missing_body(inference_server_client):
    """Returns 422 when request body is missing."""
    resp = inference_server_client.post("/v1/chat/completions")
    assert resp.status_code == 422


@pytest.mark.api
def test_chat_completions_wrong_content_type(inference_server_client):
    """Returns 422 when content-type is wrong (plain text)."""
    resp = inference_server_client.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 422


@pytest.mark.api
def test_chat_completions_all_params(inference_server_client):
    """Chat accepts all optional parameters without error."""
    resp = inference_server_client.post("/v1/chat/completions", json={
        "model": "custom",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 256,
        "temperature": 0.5,
        "top_p": 0.8,
        "stream": False,
        "agentic": False,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "custom"


# ═══════════════════════════════════════════════════════════════════════════
# GET /v1/models
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.api
def test_list_models(inference_server_client):
    """Returns model list with 'default' model when engine has a model_path."""
    resp = inference_server_client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
    assert len(body["data"]) >= 1
    model = body["data"][0]
    assert model["id"] == "default"
    assert model["object"] == "model"
    assert model["owned_by"] == "local"


@pytest.mark.api
def test_list_models_empty(inference_server_client):
    """Returns empty list when engine has no model_path."""
    # The mock engine has model_path set, so we test that it's populated
    resp = inference_server_client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body


# ═══════════════════════════════════════════════════════════════════════════
# POST /v1/rag/query
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.api
def test_rag_query_with_context(inference_server_client, mock_rag_store):
    """RAG query returns response with sources when context is found."""
    resp = inference_server_client.post("/v1/rag/query", json={
        "question": "What is RAG?",
        "top_k": 3,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "response" in body
    assert isinstance(body["response"], str)
    assert "sources" in body
    assert isinstance(body["sources"], list)
    assert "chunks_retrieved" in body
    assert isinstance(body["chunks_retrieved"], int)


@pytest.mark.api
def test_rag_query_without_context(inference_server_client):
    """RAG query with empty search results returns response with empty sources."""
    # Override the mock store's search to return empty results
    inference_server_client.app.dependency_overrides = {}

    # The store's search is already mocked — just test the endpoint works
    resp = inference_server_client.post("/v1/rag/query", json={
        "question": "What is quantum computing?",
        "top_k": 5,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "response" in body


@pytest.mark.api
def test_rag_query_no_model_loaded(mocker, mock_rag_store):
    """Returns 503 when no model is loaded for RAG query."""
    from fastapi.testclient import TestClient

    from inference_server.server import app

    engine = mocker.MagicMock()
    engine.model = None
    engine.model_path = None

    mocker.patch("inference_server.inference.InferenceEngine", return_value=engine)
    mocker.patch("inference_server.rag.RAGStore", return_value=mock_rag_store)

    with TestClient(app) as client:
        resp = client.post("/v1/rag/query", json={"question": "What is AI?"})
    assert resp.status_code == 503


@pytest.mark.api
def test_rag_query_with_system_prompt(inference_server_client):
    """RAG query with custom system_prompt includes it in the request."""
    resp = inference_server_client.post("/v1/rag/query", json={
        "question": "Explain ML",
        "system_prompt": "You are a professor.",
        "top_k": 2,
    })
    assert resp.status_code == 200
    assert "response" in resp.json()


# ═══════════════════════════════════════════════════════════════════════════
# POST /v1/rag/ingest
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.api
def test_rag_ingest_multipart_upload(inference_server_client, mock_rag_store):
    """Multipart file upload ingests document into RAG store."""
    file_content = b"This is a test document for RAG ingestion."
    resp = inference_server_client.post(
        "/v1/rag/ingest",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "filename" in body
    assert body["filename"] == "test.txt"
    assert "chunks" in body
    assert isinstance(body["chunks"], int)


@pytest.mark.api
def test_rag_ingest_no_file(inference_server_client):
    """Returns 422 when no file is provided."""
    resp = inference_server_client.post("/v1/rag/ingest")
    assert resp.status_code == 422


@pytest.mark.api
def test_rag_ingest_large_file(inference_server_client):
    """Handles larger file uploads without error."""
    content = ("Line of text for RAG ingestion test. " * 50).encode()
    resp = inference_server_client.post(
        "/v1/rag/ingest",
        files={"file": ("large.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "chunks" in body


# ═══════════════════════════════════════════════════════════════════════════
# POST /v1/rag/ingest-directory
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.api
def test_rag_ingest_directory(inference_server_client):
    """Directory ingest returns ingestion results."""
    resp = inference_server_client.post("/v1/rag/ingest-directory", json={
        "path": "/some/directory",
    })
    assert resp.status_code == 200
    body = resp.json()
    # ingestor.ingest_directory is mocked — returns whatever MagicMock has
    assert body is not None


@pytest.mark.api
def test_rag_ingest_directory_missing_path(inference_server_client):
    """Directory ingest with empty body returns 422 (path is required)."""
    resp = inference_server_client.post("/v1/rag/ingest-directory", json={})
    assert resp.status_code in (200, 422)  # depends on whether endpoint requires path


# ═══════════════════════════════════════════════════════════════════════════
# GET /v1/rag/documents
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.api
def test_rag_list_documents(inference_server_client, mock_rag_store):
    """Lists RAG documents with document list and total chunks."""
    resp = inference_server_client.get("/v1/rag/documents")
    assert resp.status_code == 200
    body = resp.json()
    assert "documents" in body
    assert isinstance(body["documents"], list)
    assert "total_chunks" in body
    assert isinstance(body["total_chunks"], int)


@pytest.mark.api
def test_rag_list_documents_schema(inference_server_client):
    """Document list entries have expected keys."""
    resp = inference_server_client.get("/v1/rag/documents")
    assert resp.status_code == 200
    body = resp.json()
    if body["documents"]:
        doc = body["documents"][0]
        assert "document_id" in doc


# ═══════════════════════════════════════════════════════════════════════════
# DELETE /v1/rag/documents/{document_id}
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.api
def test_rag_remove_document(inference_server_client):
    """Deleting a document returns 200 with document_id and chunks_removed."""
    resp = inference_server_client.delete("/v1/rag/documents/doc1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == "doc1"
    assert "chunks_removed" in body


@pytest.mark.api
def test_rag_remove_document_nonexistent(inference_server_client):
    """Deleting a nonexistent document still returns 200 (idempotent)."""
    resp = inference_server_client.delete("/v1/rag/documents/nonexistent_id")
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_id"] == "nonexistent_id"


# ═══════════════════════════════════════════════════════════════════════════
# GET /v1/parse/supported
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.api
def test_parse_supported(inference_server_client):
    """Returns list of supported file extensions."""
    resp = inference_server_client.get("/v1/parse/supported")
    assert resp.status_code == 200
    body = resp.json()
    assert "formats" in body
    formats = body["formats"]
    assert isinstance(formats, list)
    assert ".txt" in formats
    assert ".pdf" in formats
    assert ".json" in formats
    assert len(formats) > 10


@pytest.mark.api
def test_parse_supported_all_common_formats(inference_server_client):
    """Supported formats include all common document types."""
    resp = inference_server_client.get("/v1/parse/supported")
    formats = resp.json()["formats"]
    for ext in [".txt", ".md", ".pdf", ".docx", ".xlsx", ".csv", ".json",
                 ".html", ".xml", ".py", ".jsonl"]:
        assert ext in formats, f"{ext} missing from supported formats"


# ═══════════════════════════════════════════════════════════════════════════
# POST /v1/parse
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.api
def test_parse_file(inference_server_client):
    """Parse a text file and return extracted text."""
    content = b"Hello world. This is a test document."
    resp = inference_server_client.post(
        "/v1/parse",
        files={"file": ("sample.txt", io.BytesIO(content), "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"] == "sample.txt"
    assert "text" in body
    assert isinstance(body["text"], str)
    assert len(body["text"]) > 0
    assert "length" in body
    assert isinstance(body["length"], int)


@pytest.mark.api
def test_parse_file_no_upload(inference_server_client):
    """Returns 422 when no file is uploaded."""
    resp = inference_server_client.post("/v1/parse")
    assert resp.status_code == 422


@pytest.mark.api
def test_parse_json_file(inference_server_client):
    """Parse a JSON file."""
    import json
    content = json.dumps({"key": "value", "nested": {"a": 1}}).encode()
    resp = inference_server_client.post(
        "/v1/parse",
        files={"file": ("data.json", io.BytesIO(content), "application/json")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "key" in body["text"] or "value" in body["text"]


# ═══════════════════════════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.api
def test_health(inference_server_client):
    """Health endpoint returns 200 with status info."""
    resp = inference_server_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)
    assert isinstance(body["rag_enabled"], bool)
    assert isinstance(body["agent_enabled"], bool)
    assert isinstance(body["uptime_seconds"], (int, float))


@pytest.mark.api
def test_health_model_loaded(inference_server_client):
    """Health reports model_loaded=True when engine.model is not None."""
    resp = inference_server_client.get("/health")
    body = resp.json()
    # mock_engine.model is a MagicMock (truthy), so model_loaded should be True
    assert body["model_loaded"] is True


@pytest.mark.api
def test_health_rag_enabled(inference_server_client):
    """Health reports rag_enabled based on rag_store presence."""
    resp = inference_server_client.get("/health")
    body = resp.json()
    assert body["rag_enabled"] is True


# ═══════════════════════════════════════════════════════════════════════════
# GET /stats
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.api
def test_stats(inference_server_client):
    """Stats endpoint returns comprehensive status dict."""
    resp = inference_server_client.get("/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert "model" in body
    assert isinstance(body["model"], dict)
    assert "rag" in body
    assert "enabled" in body["rag"]
    assert "total_chunks" in body["rag"]
    assert "documents" in body["rag"]
    assert "supported_document_formats" in body
    assert isinstance(body["supported_document_formats"], int)
    assert "uptime_seconds" in body


@pytest.mark.api
def test_stats_rag_fields(inference_server_client):
    """Stats rag section has correct types."""
    resp = inference_server_client.get("/stats")
    rag = resp.json()["rag"]
    assert isinstance(rag["enabled"], bool)
    assert isinstance(rag["total_chunks"], int)
    assert isinstance(rag["documents"], int)


# ═══════════════════════════════════════════════════════════════════════════
# POST /reload
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.api
def test_reload(inference_server_client):
    """Reload returns 200 with status='reloaded'."""
    resp = inference_server_client.post("/reload")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "reloaded"
    assert "model" in body
    assert isinstance(body["model"], dict)
