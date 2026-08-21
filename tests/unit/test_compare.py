"""Tests for finetune_studio.compare package.

Covers: engine.py, scorer.py, reporter.py.
"""
from unittest.mock import MagicMock, patch

import pytest

# ══════════════════════════════════════════════════════════════
# scorer.py tests
# ══════════════════════════════════════════════════════════════

class TestScorer:
    """Tests for Scorer class."""

    def test_init_defaults(self):
        """Scorer has default weights."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        assert s.keyword_weight == 0.5
        assert s.length_weight == 0.2
        assert s.time_weight == 0.1
        assert s.forbidden_weight == 0.2

    def test_init_custom_weights(self):
        """Scorer accepts custom weights."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer(keyword_weight=0.7, length_weight=0.1)
        assert s.keyword_weight == 0.7

    def test_score_keyword_match_all(self):
        """All keywords present gives score 1.0."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        kw, fp = s.score_keyword_match("Python is great", ["python", "great"], [])
        assert kw == 1.0
        assert fp == 0.0

    def test_score_keyword_match_partial(self):
        """Partial keywords gives partial score."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        kw, fp = s.score_keyword_match("Python is cool", ["python", "great"], [])
        assert kw == pytest.approx(0.5)

    def test_score_keyword_match_none(self):
        """No expected keywords gives 1.0."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        kw, fp = s.score_keyword_match("Any text", [], [])
        assert kw == 1.0

    def test_score_forbidden_penalty(self):
        """Forbidden words increase penalty."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        kw, fp = s.score_keyword_match("This is terrible", [], ["terrible", "awful"])
        assert fp == pytest.approx(0.5)

    def test_score_length_ideal(self):
        """Ideal length gets score 1.0."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        score = s.score_length("A" * 200, ideal_length=200)
        assert score == 1.0

    def test_score_length_too_short(self):
        """Very short response gets low score."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        score = s.score_length("hi", ideal_length=200)
        assert score == 0.2

    def test_score_length_empty(self):
        """Empty response gets 0."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        score = s.score_length("", ideal_length=200)
        assert score == 0.0

    def test_score_length_too_long(self):
        """Very long response gets reduced score."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        score = s.score_length("A" * 1000, ideal_length=200)
        assert score == 0.4

    def test_score_time_fast(self):
        """Fast response gets good time score."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        score = s.score_time(500, baseline_ms=1000)
        assert score == 1.0

    def test_score_time_zero(self):
        """Zero time gets 0."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        score = s.score_time(0, baseline_ms=1000)
        assert score == 0.0

    def test_score_time_slow(self):
        """Slow response gets lower time score."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        score = s.score_time(5000, baseline_ms=1000)
        assert score == 0.5

    def test_score_response_full(self):
        """score_response returns ScoreResult with all fields."""
        from finetune_studio.compare.scorer import Scorer, ScoreResult
        s = Scorer()
        result = s.score_response(
            "test1", "model1", "Python is great for coding",
            expected=["python", "great"], forbidden=[], time_ms=500,
        )
        assert isinstance(result, ScoreResult)
        assert result.test_name == "test1"
        assert result.source_name == "model1"
        assert result.keyword_score > 0
        assert result.total_score > 0

    def test_score_response_with_forbidden(self):
        """Forbidden words reduce total score."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        good = s.score_response("t", "m", "Python is great", ["python"], [], 500)
        bad = s.score_response("t", "m", "Python is terrible", ["python"], ["terrible"], 500)
        assert good.total_score > bad.total_score

    def test_score_response_passed(self):
        """passed requires total >= 0.5 and no forbidden."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        result = s.score_response("t", "m", "Python is great for coding", ["python", "great"], [], 500)
        assert result.passed is True

    def test_score_comparison(self):
        """score_comparison aggregates by source."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        comparison_results = [
            {
                "name": "test1",
                "expected_keywords": ["python"],
                "forbidden_keywords": [],
                "responses": {
                    "model1": [{"response": "Python is great", "time_ms": 500, "error": None}],
                    "model2": [{"response": "I don't know", "time_ms": 1000, "error": None}],
                },
            }
        ]
        result = s.score_comparison(comparison_results)
        assert "all_scores" in result
        assert "by_source" in result
        assert "model1" in result["by_source"]
        assert result["by_source"]["model1"]["avg_score"] > 0

    def test_score_comparison_error_handling(self):
        """score_comparison handles error responses."""
        from finetune_studio.compare.scorer import Scorer
        s = Scorer()
        comparison_results = [
            {
                "name": "test1",
                "expected_keywords": [],
                "forbidden_keywords": [],
                "responses": {
                    "model1": [{"response": "", "time_ms": 0, "error": "timeout"}],
                },
            }
        ]
        result = s.score_comparison(comparison_results)
        assert len(result["all_scores"]) == 1
        assert result["all_scores"][0].total_score == 0.0


class TestScoreResult:
    """Tests for ScoreResult dataclass."""

    def test_defaults(self):
        """ScoreResult has defaults."""
        from finetune_studio.compare.scorer import ScoreResult
        sr = ScoreResult()
        assert sr.test_name == ""
        assert sr.total_score == 0.0
        assert sr.passed is False
        assert sr.details == {}


# ══════════════════════════════════════════════════════════════
# reporter.py tests
# ══════════════════════════════════════════════════════════════

class TestReporter:
    """Tests for reporter module functions."""

    def _make_scored(self):
        from finetune_studio.compare.scorer import ScoreResult
        return {
            "by_source": {
                "model1": {
                    "pass_rate": 80.0,
                    "passed": 4,
                    "total": 5,
                    "avg_score": 0.75,
                    "avg_time_ms": 500.0,
                },
            },
            "all_scores": [
                ScoreResult(
                    test_name="test1", source_name="model1",
                    response="Python is great", keyword_score=1.0,
                    length_score=1.0, time_ms=500, total_score=0.8,
                    passed=True, details={"keyword_misses": [], "forbidden_hits": []},
                ),
            ],
        }

    def test_generate_report(self):
        """generate_report returns formatted text."""
        from finetune_studio.compare.reporter import generate_report
        comparison_results = [
            {
                "name": "test1",
                "responses": {
                    "model1": [{"response": "Python is great", "time_ms": 500, "error": None}],
                },
            }
        ]
        scored = self._make_scored()
        report = generate_report(comparison_results, scored)
        assert "COMPARISON REPORT" in report
        assert "model1" in report
        assert "Pass rate" in report

    def test_generate_report_with_output_path(self, tmp_dir):
        """generate_report writes to file when path provided."""
        from finetune_studio.compare.reporter import generate_report
        comparison_results = [
            {
                "name": "test1",
                "responses": {
                    "model1": [{"response": "Hello", "time_ms": 100, "error": None}],
                },
            }
        ]
        scored = self._make_scored()
        out = tmp_dir / "report.txt"
        report = generate_report(comparison_results, scored, output_path=str(out))
        assert out.exists()
        assert out.read_text() == report

    def test_generate_json_report(self):
        """generate_json_report returns serializable dict."""
        from finetune_studio.compare.reporter import generate_json_report
        comparison_results = [
            {
                "name": "test1",
                "responses": {
                    "model1": [{"response": "Hello", "time_ms": 100, "error": None}],
                },
            }
        ]
        scored = self._make_scored()
        json_report = generate_json_report(comparison_results, scored)
        assert "generated" in json_report
        assert "summary" in json_report
        assert "results" in json_report
        assert "model1" in json_report["summary"]

    def test_report_includes_detailed_results(self):
        """Report includes per-test detail."""
        from finetune_studio.compare.reporter import generate_report
        comparison_results = [
            {
                "name": "my_test",
                "responses": {
                    "model1": [{"response": "answer", "time_ms": 100, "error": None}],
                },
            }
        ]
        scored = self._make_scored()
        report = generate_report(comparison_results, scored)
        assert "my_test" in report

    def test_report_error_response(self):
        """Report handles error responses."""
        from finetune_studio.compare.reporter import generate_report
        comparison_results = [
            {
                "name": "test1",
                "responses": {
                    "model1": [{"response": "", "time_ms": 0, "error": "timeout"}],
                },
            }
        ]
        scored = self._make_scored()
        report = generate_report(comparison_results, scored)
        assert "ERROR" in report
        assert "timeout" in report


# ══════════════════════════════════════════════════════════════
# engine.py tests
# ══════════════════════════════════════════════════════════════

class TestComparisonConfig:
    """Tests for ComparisonConfig dataclass."""

    def test_defaults(self):
        """ComparisonConfig has defaults."""
        from finetune_studio.compare.engine import ComparisonConfig
        c = ComparisonConfig()
        assert c.max_tokens == 512
        assert c.temperature == 0.7
        assert c.runs_per_prompt == 1
        assert c.timeout == 60


class TestModelSource:
    """Tests for ModelSource dataclass."""

    def test_local_source(self):
        """Local model source has path."""
        from finetune_studio.compare.engine import ModelSource
        ms = ModelSource(name="m1", type="local", path="/model.gguf")
        assert ms.type == "local"
        assert ms.path == "/model.gguf"

    def test_api_source(self):
        """API model source has url and key."""
        from finetune_studio.compare.engine import ModelSource
        ms = ModelSource(name="m1", type="api", api_url="https://api.example.com", api_key="sk-123")
        assert ms.type == "api"
        assert ms.api_url == "https://api.example.com"


@pytest.mark.slow  # needs requests + comparison engine
class TestComparisonEngine:
    """Tests for ComparisonEngine class."""

    def test_init(self):
        """ComparisonEngine initializes with config."""
        from finetune_studio.compare.engine import ComparisonConfig, ComparisonEngine
        cfg = ComparisonConfig(max_tokens=256)
        ce = ComparisonEngine(config=cfg)
        assert ce.config.max_tokens == 256

    def test_generate_response_local(self):
        """generate_response with local source calls engine."""
        from finetune_studio.compare.engine import ComparisonEngine, ModelSource
        ce = ComparisonEngine()
        mock_engine = MagicMock()
        mock_engine.generate.return_value = "Hello"
        ce._local_engines["/model"] = mock_engine
        source = ModelSource(name="m", type="local", path="/model")
        result = ce.generate_response(source, [{"role": "user", "content": "Hi"}])
        assert result["response"] == "Hello"
        assert result["error"] is None

    def test_generate_response_api(self):
        """generate_response with API source calls requests."""
        from finetune_studio.compare.engine import ComparisonEngine, ModelSource
        ce = ComparisonEngine()
        source = ModelSource(name="m", type="api", api_url="https://api.example.com/v1/chat",
                             api_key="sk-123", model_id="gpt-4")
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "API response"}}]}
        mock_response.raise_for_status = MagicMock()
        with patch("finetune_studio.compare.engine.requests.post", return_value=mock_response):
            result = ce.generate_response(source, [{"role": "user", "content": "Hi"}])
            assert result["response"] == "API response"

    def test_generate_response_error(self):
        """generate_response handles errors gracefully."""
        from finetune_studio.compare.engine import ComparisonEngine, ModelSource
        ce = ComparisonEngine()
        mock_engine = MagicMock()
        mock_engine.generate.side_effect = RuntimeError("OOM")
        ce._local_engines["/model"] = mock_engine
        source = ModelSource(name="m", type="local", path="/model")
        result = ce.generate_response(source, [{"role": "user", "content": "Hi"}])
        assert result["error"] == "OOM"
        assert result["response"] == ""

    def test_run_comparison(self):
        """run_comparison runs all tests across sources."""
        from finetune_studio.compare.engine import ComparisonEngine, ModelSource
        ce = ComparisonEngine()
        mock_engine = MagicMock()
        mock_engine.generate.return_value = {"response": "OK", "tokens": 5, "time_ms": 10}
        ce._local_engines["/m"] = mock_engine
        source = ModelSource(name="model1", type="local", path="/m")
        test_suite = [
            {"name": "test1", "messages": [{"role": "user", "content": "Q1"}],
             "expected_keywords": ["ok"], "forbidden_keywords": []},
            {"name": "test2", "messages": [{"role": "user", "content": "Q2"}],
             "expected_keywords": [], "forbidden_keywords": []},
        ]
        results = ce.run_comparison([source], test_suite)
        assert len(results) == 2
        assert "model1" in results[0]["responses"]

    def test_run_comparison_multiple_runs(self):
        """run_comparison with runs_per_prompt > 1."""
        from finetune_studio.compare.engine import ComparisonConfig, ComparisonEngine, ModelSource
        cfg = ComparisonConfig(runs_per_prompt=3)
        ce = ComparisonEngine(config=cfg)
        mock_engine = MagicMock()
        mock_engine.generate.return_value = {"response": "OK", "tokens": 5, "time_ms": 10}
        ce._local_engines["/m"] = mock_engine
        source = ModelSource(name="model1", type="local", path="/m")
        test_suite = [{"name": "t", "messages": [{"role": "user", "content": "Q"}]}]
        results = ce.run_comparison([source], test_suite)
        assert len(results[0]["responses"]["model1"]) == 3

    def test_cleanup(self):
        """cleanup unloads all engines."""
        from finetune_studio.compare.engine import ComparisonEngine
        ce = ComparisonEngine()
        mock_engine = MagicMock()
        ce._local_engines["/m"] = mock_engine
        ce.cleanup()
        mock_engine.unload.assert_called_once()
        assert ce._local_engines == {}
