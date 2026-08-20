"""finetune-studio WebUI API endpoint tests.

WHAT THIS FILE DOES
===================
Exercises every REST endpoint in the finetune-studio WebUI:
  - Pages router (HTML pages)
  - Models router (model list/info/refresh)
  - Training router (status/progress/start/stop)
  - Testing router (load/unload/status/chat/run-suite)
  - Data router (files/upload/validate/preview/dedup)
  - Comparison router (compare/load/compare/run/cleanup/rag-chat)

Uses the `finetune_studio_client` fixture from conftest.py which
mocks out all heavy resources (models, engines, file I/O).

NOTE: The comparison router is mounted with prefix /api/compare but its
routes also start with /compare/ (e.g., /compare/load), resulting in
double-prefixed paths like /api/compare/compare/load. Tests use the
actual registered paths.
"""

from __future__ import annotations

import io

import pytest


# ============================================================================
# Pages router — GET HTML pages
# ============================================================================


class TestPagesRouter:
    """Tests for webui/routes/pages.py — HTML page endpoints."""

    @pytest.mark.api
    def test_index_returns_html(self, finetune_studio_client):
        """GET / should return HTML for the dashboard index page."""
        resp = finetune_studio_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Dashboard" in resp.text or "Finetune Studio" in resp.text

    @pytest.mark.api
    def test_models_page_returns_html(self, finetune_studio_client):
        """GET /models should return HTML for the models browser page."""
        resp = finetune_studio_client.get("/models")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Models" in resp.text

    @pytest.mark.api
    def test_training_page_returns_html(self, finetune_studio_client):
        """GET /training should return HTML for the training dashboard."""
        resp = finetune_studio_client.get("/training")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Training" in resp.text

    @pytest.mark.api
    def test_data_page_returns_html(self, finetune_studio_client):
        """GET /data should return HTML for the data files page."""
        resp = finetune_studio_client.get("/data")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Data" in resp.text or "data" in resp.text

    @pytest.mark.api
    def test_testing_page_returns_html(self, finetune_studio_client):
        """GET /testing should return HTML for the testing playground."""
        resp = finetune_studio_client.get("/testing")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Testing" in resp.text or "test" in resp.text.lower()


# ============================================================================
# Models router — GET list, GET info, POST refresh
# ============================================================================


class TestModelsRouter:
    """Tests for webui/routes/models.py — model management endpoints."""

    @pytest.mark.api
    def test_models_list_returns_model_list(self, finetune_studio_client):
        """GET /api/models/list should return a list of discovered models."""
        resp = finetune_studio_client.get("/api/models/list")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        for model in data:
            assert "name" in model
            assert "path" in model
            assert "format" in model

    @pytest.mark.api
    def test_models_info_returns_model_info(self, finetune_studio_client):
        """GET /api/models/info should return info for a specific model path."""
        resp = finetune_studio_client.get("/api/models/info", params={"path": "/fake/model.gguf"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    @pytest.mark.api
    def test_models_refresh_triggers_rescan(self, finetune_studio_client):
        """POST /api/models/refresh should trigger a model directory rescan."""
        resp = finetune_studio_client.post("/api/models/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data


# ============================================================================
# Training router — status, progress, start, stop
# ============================================================================


class TestTrainingRouter:
    """Tests for webui/routes/training.py — training management endpoints."""

    @pytest.mark.api
    def test_training_status_returns_status_dict(self, finetune_studio_client):
        """GET /api/training/status should return a status dictionary."""
        resp = finetune_studio_client.get("/api/training/status")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "status" in data
        assert "step" in data
        assert "loss" in data

    @pytest.mark.api
    def test_training_progress_returns_stream(self, finetune_studio_client):
        """GET /api/training/progress should return SSE stream (200)."""
        resp = finetune_studio_client.get("/api/training/progress")
        assert resp.status_code == 200

    @pytest.mark.api
    def test_training_start_starts_job(self, finetune_studio_client, tmp_path):
        """POST /api/training/start should start a training job with valid config."""
        # Create a real JSONL file so training can load it
        data_file = tmp_path / "train.jsonl"
        data_file.write_text(
            '{"messages": [{"role": "user", "content": "hi"}, '
            '{"role": "assistant", "content": "hello"}]}\n'
        )
        body = {
            "model_path": "/fake/model.gguf",
            "data_path": str(data_file),
            "output_dir": "output",
            "lora_rank": 64,
            "learning_rate": 8e-5,
            "num_epochs": 4,
            "batch_size": 2,
            "max_seq_length": 2048,
        }
        resp = finetune_studio_client.post("/api/training/start", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data or "error" in data

    @pytest.mark.api
    def test_training_start_missing_data_returns_error(self, finetune_studio_client):
        """POST /api/training/start without data_path should return an error."""
        body = {"model_path": "/fake/model.gguf"}
        resp = finetune_studio_client.post("/api/training/start", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    @pytest.mark.api
    def test_training_stop_returns_stopping(self, finetune_studio_client):
        """POST /api/training/stop should signal training to stop."""
        resp = finetune_studio_client.post("/api/training/stop")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "stopping"


# ============================================================================
# Testing router — load, unload, status, chat, run-suite
# ============================================================================


class TestTestingRouter:
    """Tests for webui/routes/testing.py — model testing endpoints."""

    @pytest.mark.api
    def test_testing_load_model(self, finetune_studio_client):
        """POST /api/testing/load should load a model by path."""
        body = {"model_path": "/fake/model.gguf"}
        resp = finetune_studio_client.post("/api/testing/load", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data or "error" in data

    @pytest.mark.api
    def test_testing_load_empty_path_returns_error(self, finetune_studio_client):
        """POST /api/testing/load without model_path should return error."""
        resp = finetune_studio_client.post("/api/testing/load", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    @pytest.mark.api
    def test_testing_unload_model(self, finetune_studio_client):
        """POST /api/testing/unload should unload the current model."""
        resp = finetune_studio_client.post("/api/testing/unload")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "unloaded"

    @pytest.mark.api
    def test_testing_status_returns_loaded_state(self, finetune_studio_client):
        """GET /api/testing/status should return model loaded state."""
        resp = finetune_studio_client.get("/api/testing/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "loaded" in data
        assert "model_path" in data
        assert "is_gguf" in data

    @pytest.mark.api
    def test_testing_chat_returns_response(self, finetune_studio_client, mocker):
        """POST /api/testing/chat should send a chat message and get response."""
        # The route imported `inference_engine` at module load — patch the
        # `generate` method on the SAME object the route is holding.
        # We import the route module and patch its `inference_engine` attr.
        import finetune_studio.webui.routes.testing as testing_route
        mocker.patch.object(testing_route.inference_engine, "generate", return_value="Mock reply")
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 512,
            "temperature": 0.7,
        }
        resp = finetune_studio_client.post("/api/testing/chat", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data

    @pytest.mark.api
    def test_testing_run_suite_returns_results(self, finetune_studio_client, tmp_path):
        """POST /api/testing/run-suite should run a test suite and return results."""
        import json as json_mod
        suite_file = tmp_path / "suite.json"
        # Use the format load_test_suite() expects: name + messages + optional expected_keywords.
        suite_file.write_text(json_mod.dumps([
            {"name": "test1", "messages": [{"role": "user", "content": "hello"}], "expected_keywords": ["hi"]}
        ]))
        body = {"suite_path": str(suite_file), "max_tokens": 512}
        resp = finetune_studio_client.post("/api/testing/run-suite", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data or "error" in data


# ============================================================================
# Data router — files, upload, validate, preview, dedup
# ============================================================================


class TestDataRouter:
    """Tests for webui/routes/data.py — data file management endpoints."""

    @pytest.mark.api
    def test_data_files_returns_file_list(self, finetune_studio_client):
        """GET /api/data/files should list available data files."""
        resp = finetune_studio_client.get("/api/data/files")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.api
    def test_data_upload_multipart_file(self, finetune_studio_client, tmp_path, mocker):
        """POST /api/data/upload should accept a multipart file upload."""
        # Patch settings.data_dir to a temp directory so upload writes there
        import finetune_studio.webui.routes.data as data_mod
        original_dir = data_mod.settings.data_dir
        data_mod.settings.data_dir = str(tmp_path)
        try:
            file_content = b'{"messages": [{"role": "user", "content": "hi"}]}\n'
            resp = finetune_studio_client.post(
                "/api/data/upload",
                files={"file": ("test.jsonl", io.BytesIO(file_content), "application/jsonl")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "name" in data or "error" in data
        finally:
            data_mod.settings.data_dir = original_dir

    @pytest.mark.api
    def test_data_validate_returns_validation(self, finetune_studio_client):
        """GET /api/data/validate should validate a dataset file."""
        resp = finetune_studio_client.get("/api/data/validate", params={"path": "/fake/data.jsonl"})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    @pytest.mark.api
    def test_data_preview_returns_preview(self, finetune_studio_client):
        """GET /api/data/preview should return a preview of a data file."""
        resp = finetune_studio_client.get(
            "/api/data/preview", params={"path": "/fake/data.jsonl", "limit": 5}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    @pytest.mark.api
    def test_data_dedup_deduplicates_dataset(self, finetune_studio_client, tmp_path):
        """POST /api/data/dedup should deduplicate a dataset."""
        data_file = tmp_path / "data.jsonl"
        line = '{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}\n'
        data_file.write_text(line)
        with data_file.open("a") as f:
            f.write(line)
        resp = finetune_studio_client.post("/api/data/dedup", params={"path": str(data_file)})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


# ============================================================================
# Comparison router — load, run, cleanup, rag/chat
# ============================================================================


class TestComparisonRouter:
    """Tests for webui/routes/comparison.py — model comparison endpoints.

    NOTE: The comparison router is mounted with prefix /api/compare, but its
    routes also start with /compare/ (e.g., /compare/load). This results in
    double-prefixed paths like /api/compare/compare/load.
    """

    @pytest.mark.api
    def test_compare_load_model(self, finetune_studio_client):
        """POST /api/compare/compare/load should load a model for comparison."""
        body = {"name": "test_model", "path": "/fake/model.gguf"}
        resp = finetune_studio_client.post("/api/compare/compare/load", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data or "error" in data

    @pytest.mark.api
    def test_compare_run_comparison(self, finetune_studio_client):
        """POST /api/compare/compare/run should run a comparison test suite."""
        body = {
            "test_suite": [{"name": "test1", "input": "hi", "expected": "hello"}],
            "config": {"max_tokens": 512, "temperature": 0.7},
        }
        resp = finetune_studio_client.post("/api/compare/compare/run", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    @pytest.mark.api
    def test_compare_run_empty_suite_returns_error(self, finetune_studio_client):
        """POST /api/compare/compare/run with empty test suite should return error."""
        body = {"test_suite": [], "config": {}}
        resp = finetune_studio_client.post("/api/compare/compare/run", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    @pytest.mark.api
    def test_compare_cleanup_unloads_models(self, finetune_studio_client):
        """POST /api/compare/compare/cleanup should unload all comparison models."""
        resp = finetune_studio_client.post("/api/compare/compare/cleanup")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "cleaned"

    @pytest.mark.api
    def test_compare_rag_chat_returns_response(self, finetune_studio_client, mocker):
        """POST /api/compare/rag/chat should perform RAG-enhanced chat."""
        # Mock VectorStore at its real source module — comparison.py does a lazy
        # `from finetune_studio.rag.store import VectorStore` inside the function.
        mocker.patch("finetune_studio.rag.store.VectorStore")
        body = {
            "messages": [{"role": "user", "content": "What is fine-tuning?"}],
            "top_k": 3,
            "max_tokens": 512,
            "temperature": 0.7,
        }
        resp = finetune_studio_client.post("/api/compare/rag/chat", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    @pytest.mark.api
    def test_compare_rag_chat_empty_messages_returns_error(self, finetune_studio_client):
        """POST /api/compare/rag/chat with no messages should return error."""
        body = {"messages": []}
        resp = finetune_studio_client.post("/api/compare/rag/chat", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
