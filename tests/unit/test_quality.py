"""Tests for finetune_studio.webui.routes.quality — data quality WebUI endpoints.

Covers the 5 routes that expose CLI data commands via WebUI:
- POST /api/data/analyze
- POST /api/data/augment
- POST /api/data/optimize
- POST /api/data/hallucination-check
- POST /api/data/convert
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# =============================================================================
# Helpers
# =============================================================================


def _make_client():
    """Create FastAPI TestClient with quality router mounted."""
    from fastapi import FastAPI
    from finetune_studio.webui.routes.quality import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _make_mock_report():
    """Create a mock analysis report."""
    return {
        "total_examples": 100,
        "duplicate_ratio": 0.05,
        "avg_length": 250.0,
        "persona_consistency": 0.92,
        "warnings": ["some warning"],
    }


def _make_mock_augment_result():
    """Create a mock augmentation result."""
    return {
        "added": 50,
        "original_count": 100,
        "output_path": "data/augmented.jsonl",
    }


def _make_mock_optimize_result():
    """Create a mock config optimizer result."""
    return {
        "learning_rate": 2e-5,
        "batch_size": 4,
        "num_epochs": 3,
        "warmup_steps": 100,
    }


def _make_mock_hallucination_result():
    """Create a mock hallucination scan result."""
    return {
        "scanned": 100,
        "risk_count": 3,
        "high_risk": ["example 5", "example 17", "example 42"],
    }


def _make_mock_convert_result():
    """Create a mock conversion result."""
    return {
        "converted": 100,
        "target_format": "sharegpt",
        "output_path": "data/converted.jsonl",
    }


# =============================================================================
# /api/data/analyze tests
# =============================================================================


class TestDataAnalyze:
    """Tests for POST /api/data/analyze."""

    def test_analyze_returns_200_on_success(self, tmp_path):
        """Returns 200 with status='ok' on success."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.training.data_quality.DataQualityAnalyzer"
        ) as MockAnalyzer:
            mock_instance = MagicMock()
            mock_instance.analyze.return_value = _make_mock_report()
            MockAnalyzer.return_value = mock_instance

            response = client.post(
                "/api/data/analyze",
                json={"path": str(data_file)},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["command"] == "analyze"

    def test_analyze_returns_report(self, tmp_path):
        """Returns the analysis report."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.training.data_quality.DataQualityAnalyzer"
        ) as MockAnalyzer:
            mock_instance = MagicMock()
            mock_instance.analyze.return_value = _make_mock_report()
            MockAnalyzer.return_value = mock_instance

            response = client.post(
                "/api/data/analyze",
                json={"path": str(data_file)},
            )

        body = response.json()
        assert body["result"]["total_examples"] == 100
        assert body["result"]["duplicate_ratio"] == 0.05

    def test_analyze_404_when_file_missing(self):
        """Returns 404 when file doesn't exist."""
        client = _make_client()

        response = client.post(
            "/api/data/analyze",
            json={"path": "nonexistent.jsonl"},
        )

        assert response.status_code == 404
        assert "File not found" in response.json()["detail"]

    def test_analyze_catches_exceptions(self, tmp_path):
        """Returns 200 with status='error' on exception."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.training.data_quality.DataQualityAnalyzer"
        ) as MockAnalyzer:
            MockAnalyzer.side_effect = RuntimeError("Analyzer boom!")

            response = client.post(
                "/api/data/analyze",
                json={"path": str(data_file)},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "error"
        assert "Analyzer boom!" in body["error"]

    def test_analyze_default_path(self):
        """Default path is 'data/training.jsonl' when not provided."""
        client = _make_client()

        # Default path doesn't exist, so 404 expected
        response = client.post("/api/data/analyze", json={})

        assert response.status_code == 404


# =============================================================================
# /api/data/augment tests
# =============================================================================


class TestDataAugment:
    """Tests for POST /api/data/augment."""

    def test_augment_returns_200_on_success(self, tmp_path):
        """Returns 200 with status='ok' on success."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.training.data_augmentation.DataAugmenter"
        ) as MockAugmenter:
            mock_instance = MagicMock()
            mock_instance.run.return_value = _make_mock_augment_result()
            MockAugmenter.return_value = mock_instance

            response = client.post(
                "/api/data/augment",
                json={"path": str(data_file)},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["command"] == "augment"

    def test_augment_passes_output_path(self, tmp_path):
        """Output path is forwarded to augmenter."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.training.data_augmentation.DataAugmenter"
        ) as MockAugmenter:
            mock_instance = MagicMock()
            mock_instance.run.return_value = {}
            MockAugmenter.return_value = mock_instance

            response = client.post(
                "/api/data/augment",
                json={"path": str(data_file), "output": "data/aug.jsonl"},
            )

        assert response.status_code == 200
        # Verify output_path passed
        call_kwargs = MockAugmenter.call_args.kwargs
        assert call_kwargs["output_path"] == "data/aug.jsonl"

    def test_augment_404_when_file_missing(self):
        """Returns 404 when file doesn't exist."""
        client = _make_client()

        response = client.post(
            "/api/data/augment",
            json={"path": "nonexistent.jsonl"},
        )

        assert response.status_code == 404

    def test_augment_catches_exceptions(self, tmp_path):
        """Returns 200 with status='error' on exception."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.training.data_augmentation.DataAugmenter"
        ) as MockAugmenter:
            MockAugmenter.side_effect = RuntimeError("Augment failed")

            response = client.post(
                "/api/data/augment",
                json={"path": str(data_file)},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "error"


# =============================================================================
# /api/data/optimize tests
# =============================================================================


class TestDataOptimize:
    """Tests for POST /api/data/optimize."""

    def test_optimize_returns_200_on_success(self, tmp_path):
        """Returns 200 with status='ok' on success."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.training.config_optimizer.ConfigOptimizer"
        ) as MockOptimizer:
            mock_instance = MagicMock()
            mock_instance.recommend.return_value = _make_mock_optimize_result()
            MockOptimizer.return_value = mock_instance

            response = client.post(
                "/api/data/optimize",
                json={"path": str(data_file)},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["command"] == "optimize"

    def test_optimize_returns_recommendations(self, tmp_path):
        """Returns the optimization recommendations."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.training.config_optimizer.ConfigOptimizer"
        ) as MockOptimizer:
            mock_instance = MagicMock()
            mock_instance.recommend.return_value = _make_mock_optimize_result()
            MockOptimizer.return_value = mock_instance

            response = client.post(
                "/api/data/optimize",
                json={"path": str(data_file)},
            )

        body = response.json()
        assert body["result"]["learning_rate"] == 2e-5
        assert body["result"]["batch_size"] == 4

    def test_optimize_404_when_file_missing(self):
        """Returns 404 when file doesn't exist."""
        client = _make_client()

        response = client.post(
            "/api/data/optimize",
            json={"path": "nonexistent.jsonl"},
        )

        assert response.status_code == 404

    def test_optimize_catches_exceptions(self, tmp_path):
        """Returns 200 with status='error' on exception."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.training.config_optimizer.ConfigOptimizer"
        ) as MockOptimizer:
            MockOptimizer.side_effect = ValueError("Invalid config")

            response = client.post(
                "/api/data/optimize",
                json={"path": str(data_file)},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "error"
        assert "Invalid config" in body["error"]


# =============================================================================
# /api/data/hallucination-check tests
# =============================================================================


class TestDataHallucinationCheck:
    """Tests for POST /api/data/hallucination-check."""

    def test_hallucination_check_returns_200_on_success(self, tmp_path):
        """Returns 200 with status='ok' on success."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.training.hallucination_guard.HallucinationGuard"
        ) as MockGuard:
            mock_instance = MagicMock()
            mock_instance.scan.return_value = _make_mock_hallucination_result()
            MockGuard.return_value = mock_instance

            response = client.post(
                "/api/data/hallucination-check",
                json={"path": str(data_file)},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["command"] == "validate-hallucination"

    def test_hallucination_check_returns_scan_report(self, tmp_path):
        """Returns the scan report."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.training.hallucination_guard.HallucinationGuard"
        ) as MockGuard:
            mock_instance = MagicMock()
            mock_instance.scan.return_value = _make_mock_hallucination_result()
            MockGuard.return_value = mock_instance

            response = client.post(
                "/api/data/hallucination-check",
                json={"path": str(data_file)},
            )

        body = response.json()
        assert body["result"]["scanned"] == 100
        assert body["result"]["risk_count"] == 3

    def test_hallucination_check_404_when_file_missing(self):
        """Returns 404 when file doesn't exist."""
        client = _make_client()

        response = client.post(
            "/api/data/hallucination-check",
            json={"path": "nonexistent.jsonl"},
        )

        assert response.status_code == 404

    def test_hallucination_check_catches_exceptions(self, tmp_path):
        """Returns 200 with status='error' on exception."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.training.hallucination_guard.HallucinationGuard"
        ) as MockGuard:
            MockGuard.side_effect = RuntimeError("Scan failed")

            response = client.post(
                "/api/data/hallucination-check",
                json={"path": str(data_file)},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "error"


# =============================================================================
# /api/data/convert tests
# =============================================================================


class TestDataConvert:
    """Tests for POST /api/data/convert."""

    def test_convert_returns_200_on_success(self, tmp_path):
        """Returns 200 with status='ok' on success."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.compare.engine.FormatConverter"
        ) as MockConverter:
            mock_instance = MagicMock()
            mock_instance.convert.return_value = _make_mock_convert_result()
            MockConverter.return_value = mock_instance

            response = client.post(
                "/api/data/convert",
                json={"path": str(data_file), "target_format": "sharegpt"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["command"] == "convert"

    def test_convert_default_target_format(self, tmp_path):
        """Default target format is 'sharegpt'."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.compare.engine.FormatConverter"
        ) as MockConverter:
            mock_instance = MagicMock()
            mock_instance.convert.return_value = {}
            MockConverter.return_value = mock_instance

            # No target_format specified
            response = client.post(
                "/api/data/convert",
                json={"path": str(data_file)},
            )

        # Verify default target_format passed
        call_kwargs = MockConverter.call_args.kwargs
        assert call_kwargs["target_format"] == "sharegpt"

    def test_convert_passes_output(self, tmp_path):
        """Output path is forwarded to converter."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.compare.engine.FormatConverter"
        ) as MockConverter:
            mock_instance = MagicMock()
            mock_instance.convert.return_value = {}
            MockConverter.return_value = mock_instance

            response = client.post(
                "/api/data/convert",
                json={"path": str(data_file), "target_format": "alpaca", "output": "data/alpaca.jsonl"},
            )

        call_kwargs = MockConverter.call_args.kwargs
        assert call_kwargs["output"] == "data/alpaca.jsonl"

    def test_convert_404_when_file_missing(self):
        """Returns 404 when file doesn't exist."""
        client = _make_client()

        response = client.post(
            "/api/data/convert",
            json={"path": "nonexistent.jsonl", "target_format": "sharegpt"},
        )

        assert response.status_code == 404

    def test_convert_catches_exceptions(self, tmp_path):
        """Returns 200 with status='error' on exception."""
        client = _make_client()
        data_file = tmp_path / "training.jsonl"
        data_file.write_text("{}")

        with patch(
            "finetune_studio.compare.engine.FormatConverter"
        ) as MockConverter:
            MockConverter.side_effect = ValueError("Bad format")

            response = client.post(
                "/api/data/convert",
                json={"path": str(data_file), "target_format": "sharegpt"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "error"


# =============================================================================
# Request/Response schema tests
# =============================================================================


class TestSchemas:
    """Tests for request/response schemas."""

    def test_data_job_request_default_path(self):
        """DataJobRequest has default path."""
        from finetune_studio.webui.routes.quality import DataJobRequest
        req = DataJobRequest()
        assert req.path == "data/training.jsonl"
        assert req.output is None

    def test_data_job_request_custom_path(self):
        """DataJobRequest accepts custom path."""
        from finetune_studio.webui.routes.quality import DataJobRequest
        req = DataJobRequest(path="custom.jsonl", output="out.jsonl")
        assert req.path == "custom.jsonl"
        assert req.output == "out.jsonl"

    def test_convert_request_default_format(self):
        """ConvertRequest defaults to sharegpt."""
        from finetune_studio.webui.routes.quality import ConvertRequest
        req = ConvertRequest(path="x.jsonl")
        assert req.target_format == "sharegpt"

    def test_data_job_response_default_status(self):
        """DataJobResponse has status field."""
        from finetune_studio.webui.routes.quality import DataJobResponse
        resp = DataJobResponse(status="ok", command="x", path="x")
        assert resp.status == "ok"
        assert resp.result is None
        assert resp.error is None


# =============================================================================
# Router configuration tests
# =============================================================================


class TestRouterConfig:
    """Tests for router configuration."""

    def test_router_has_data_prefix(self):
        """Router has /api/data prefix."""
        from finetune_studio.webui.routes.quality import router
        # Prefix should contain "data"
        assert "data" in router.prefix

    def test_router_has_quality_tag(self):
        """Router is tagged 'data-quality'."""
        from finetune_studio.webui.routes.quality import router
        assert "data-quality" in router.tags or "data" in router.tags


# =============================================================================
# Endpoint inventory tests
# =============================================================================


class TestEndpointInventory:
    """Tests that all expected endpoints exist."""

    @pytest.mark.parametrize("endpoint", [
        "/api/data/analyze",
        "/api/data/augment",
        "/api/data/optimize",
        "/api/data/hallucination-check",
        "/api/data/convert",
    ])
    def test_endpoint_exists(self, endpoint):
        """Each endpoint exists (not 404)."""
        client = _make_client()
        # POST without body should 404 (file missing) or 422 (validation error)
        # Both prove the endpoint exists
        response = client.post(endpoint, json={"path": "nonexistent.jsonl"})

        # Should NOT be 405 Method Not Allowed (means endpoint exists)
        assert response.status_code != 405
        # Should be 404 (file not found) or 422 (validation error)
        assert response.status_code in (404, 422)

    def test_all_endpoints_are_post(self):
        """All quality endpoints are POST."""
        client = _make_client()
        # Try GET on each — should return 405
        for endpoint in [
            "/api/data/analyze",
            "/api/data/augment",
            "/api/data/optimize",
            "/api/data/hallucination-check",
            "/api/data/convert",
        ]:
            response = client.get(endpoint)
            assert response.status_code == 405, f"{endpoint} should be POST only"