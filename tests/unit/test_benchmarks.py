"""Tests for finetune_studio.benchmarks package.

Covers: __init__.py, comparison.py, samplers.py, scoring.py, tool_calling.py, real_benchmarks.py.
"""
import json
from unittest.mock import MagicMock, patch

import pytest


# ══════════════════════════════════════════════════════════════
# scoring.py tests
# ══════════════════════════════════════════════════════════════

class TestBenchmarkScorer:
    """Tests for BenchmarkScorer class."""

    def test_score_mcq_correct(self):
        from finetune_studio.benchmarks.scoring import scorer
        result = scorer.score_mcq("B) Madrid", "B")
        assert result["correct"] is True
        assert result["prediction"] == "B"

    def test_score_mcq_incorrect(self):
        from finetune_studio.benchmarks.scoring import scorer
        result = scorer.score_mcq("A) Berlin", "B")
        assert result["correct"] is False

    def test_score_mcq_answer_is_format(self):
        from finetune_studio.benchmarks.scoring import scorer
        result = scorer.score_mcq("The answer is C", "C")
        assert result["correct"] is True

    def test_score_mcq_no_letter(self):
        from finetune_studio.benchmarks.scoring import scorer
        result = scorer.score_mcq("I think Paris is nice", "C")
        assert result["prediction"] == ""

    def test_extract_mcq_letter_direct(self):
        from finetune_studio.benchmarks.scoring import scorer
        assert scorer.extract_mcq_letter("A) Berlin") == "A"
        assert scorer.extract_mcq_letter("B. Madrid") == "B"

    def test_extract_mcq_letter_answer_is(self):
        from finetune_studio.benchmarks.scoring import scorer
        assert scorer.extract_mcq_letter("The answer is D") == "D"

    def test_extract_mcq_letter_choice_match(self):
        from finetune_studio.benchmarks.scoring import scorer
        assert scorer.extract_mcq_letter("Paris is the capital", choices=["Berlin", "Madrid", "Paris", "Rome"]) == "C"

    def test_extract_mcq_letter_empty(self):
        from finetune_studio.benchmarks.scoring import scorer
        assert scorer.extract_mcq_letter("something random") == ""

    def test_score_math_correct(self):
        from finetune_studio.benchmarks.scoring import scorer
        result = scorer.score_math("The answer is 42", "42")
        assert result["correct"] is True

    def test_score_math_with_hash_marker(self):
        from finetune_studio.benchmarks.scoring import scorer
        result = scorer.score_math("Step 1...\n#### 120", "120")
        assert result["correct"] is True

    def test_score_math_incorrect(self):
        from finetune_studio.benchmarks.scoring import scorer
        result = scorer.score_math("The answer is 10", "42")
        assert result["correct"] is False

    def test_normalize_number(self):
        from finetune_studio.benchmarks.scoring import scorer
        assert scorer.normalize_number("1,000") == "1000.0"
        assert scorer.normalize_number("$42.50") == "42.5"
        assert scorer.normalize_number("  30  ") == "30.0"
        assert scorer.normalize_number("") == ""

    def test_score_truthful_correct(self):
        from finetune_studio.benchmarks.scoring import scorer
        result = scorer.score_truthful(
            "No, the earth is round",
            correct_answers=["no", "round", "sphere"],
            incorrect_answers=["yes", "flat"],
        )
        assert result["correct"] is True
        assert len(result["matches_correct"]) > 0
        assert len(result["matches_incorrect"]) == 0

    def test_score_truthful_incorrect(self):
        from finetune_studio.benchmarks.scoring import scorer
        result = scorer.score_truthful(
            "Yes, the earth is flat",
            correct_answers=["no", "round"],
            incorrect_answers=["yes", "flat"],
        )
        assert result["correct"] is False

    def test_score_winogrande(self):
        from finetune_studio.benchmarks.scoring import scorer
        result = scorer.score_winogrande("1", "trophy", "suitcase")
        assert result["prediction"] == "1"
        result2 = scorer.score_winogrande("The suitcase", "trophy", "suitcase")
        assert result2["prediction"] == "2"

    def test_score_open_ended_all_keywords(self):
        from finetune_studio.benchmarks.scoring import scorer
        result = scorer.score_open_ended(
            "Python is a programming language used for machine learning",
            "",
            keywords=["python", "programming", "machine learning"],
            forbidden=[],
        )
        assert result["correct"] is True
        assert result["keyword_score"] == 1.0

    def test_score_open_ended_forbidden_penalty(self):
        from finetune_studio.benchmarks.scoring import scorer
        result = scorer.score_open_ended(
            "Python is great but actually terrible and awful",
            "",
            keywords=["python"],
            forbidden=["terrible", "awful"],
        )
        assert result["forbidden_hits"] == ["terrible", "awful"]
        assert result["correct"] is False

    def test_score_open_ended_short_response(self):
        from finetune_studio.benchmarks.scoring import scorer
        result = scorer.score_open_ended("ok", "", keywords=[], forbidden=[])
        assert result["length_score"] == 0.2

    def test_score_open_ended_no_keywords(self):
        from finetune_studio.benchmarks.scoring import scorer
        result = scorer.score_open_ended("Any response", "", keywords=[], forbidden=[])
        assert result["keyword_score"] == 1.0

    def test_scorer_singleton(self):
        from finetune_studio.benchmarks.scoring import scorer, BenchmarkScorer
        assert isinstance(scorer, BenchmarkScorer)


# ══════════════════════════════════════════════════════════════
# samplers.py tests
# ══════════════════════════════════════════════════════════════

class TestSamplerConfig:
    def test_defaults(self):
        from finetune_studio.benchmarks.samplers import SamplerConfig
        s = SamplerConfig()
        assert s.temperature == 0.7
        assert s.top_p == 0.9
        assert s.top_k == 40
        assert s.repeat_penalty == 1.1
        assert s.min_p == 0.05
        assert s.max_tokens == 512
        assert s.seed == -1

    def test_to_llama_cpp_kwargs(self):
        from finetune_studio.benchmarks.samplers import SamplerConfig
        s = SamplerConfig(temperature=0.5, max_tokens=256, seed=42, stop=["<eos>"])
        kwargs = s.to_llama_cpp_kwargs()
        assert kwargs["temperature"] == 0.5
        assert kwargs["max_tokens"] == 256
        assert kwargs["seed"] == 42
        assert kwargs["stop"] == ["<eos>"]

    def test_to_llama_cpp_temperature_floor(self):
        from finetune_studio.benchmarks.samplers import SamplerConfig
        s = SamplerConfig(temperature=0.0)
        kwargs = s.to_llama_cpp_kwargs()
        assert kwargs["temperature"] == 0.01

    def test_to_llama_cpp_no_seed_when_negative(self):
        from finetune_studio.benchmarks.samplers import SamplerConfig
        s = SamplerConfig(seed=-1)
        kwargs = s.to_llama_cpp_kwargs()
        assert "seed" not in kwargs

    def test_to_openai_kwargs(self):
        from finetune_studio.benchmarks.samplers import SamplerConfig
        s = SamplerConfig(temperature=0.3, max_tokens=100, frequency_penalty=0.5)
        kwargs = s.to_openai_kwargs()
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == 100
        assert kwargs["frequency_penalty"] == 0.5

    def test_from_dict(self):
        from finetune_studio.benchmarks.samplers import SamplerConfig
        s = SamplerConfig.from_dict({"temperature": 0.9, "unknown_key": 123, "top_p": 0.8})
        assert s.temperature == 0.9
        assert s.top_p == 0.8

    def test_from_dict_empty(self):
        from finetune_studio.benchmarks.samplers import SamplerConfig
        s = SamplerConfig.from_dict({})
        assert s.temperature == 0.7


class TestSamplerPresets:
    def test_presets_exist(self):
        from finetune_studio.benchmarks.samplers import PRESETS
        assert "deterministic" in PRESETS
        assert "balanced" in PRESETS
        assert "creative" in PRESETS
        assert "testing" in PRESETS

    def test_deterministic_preset(self):
        from finetune_studio.benchmarks.samplers import PRESETS
        assert PRESETS["deterministic"].temperature == 0.0
        assert PRESETS["deterministic"].top_p == 1.0

    def test_creative_preset(self):
        from finetune_studio.benchmarks.samplers import PRESETS
        assert PRESETS["creative"].temperature == 1.0
        assert PRESETS["creative"].top_p == 0.95

    def test_get_sampler_default(self):
        from finetune_studio.benchmarks.samplers import get_sampler
        s = get_sampler()
        assert s.temperature == 0.7

    def test_get_sampler_unknown_returns_balanced(self):
        from finetune_studio.benchmarks.samplers import get_sampler
        s = get_sampler("nonexistent")
        assert s.temperature == 0.7

    def test_get_sampler_named(self):
        from finetune_studio.benchmarks.samplers import get_sampler
        s = get_sampler("deterministic")
        assert s.temperature == 0.0

    def test_list_presets(self):
        from finetune_studio.benchmarks.samplers import list_presets
        presets = list_presets()
        assert "balanced" in presets
        assert "temperature" in presets["balanced"]
        assert "top_p" in presets["balanced"]


# ══════════════════════════════════════════════════════════════
# benchmarks/__init__.py tests
# ══════════════════════════════════════════════════════════════

class TestBenchmarkResult:
    def test_benchmark_result_fields(self):
        from finetune_studio.benchmarks import BenchmarkResult
        r = BenchmarkResult(
            benchmark="test", category="cat", question="q",
            prediction="p", expected="e", correct=True,
        )
        assert r.benchmark == "test"
        assert r.correct is True
        assert r.confidence == 0.0
        assert r.time_ms == 0.0
        assert r.metadata == {}


class TestBaseBenchmark:
    def test_len(self):
        from finetune_studio.benchmarks import BaseBenchmark
        b = BaseBenchmark()
        b.samples = [{"q": "a"}, {"q": "b"}, {"q": "c"}]
        assert len(b) == 3

    def test_get_samples_all(self):
        from finetune_studio.benchmarks import BaseBenchmark
        b = BaseBenchmark()
        b.samples = [{"q": "a"}, {"q": "b"}]
        assert b.get_samples() == b.samples

    def test_get_samples_limit(self):
        from finetune_studio.benchmarks import BaseBenchmark
        b = BaseBenchmark()
        b.samples = [{"q": "a"}, {"q": "b"}, {"q": "c"}]
        assert len(b.get_samples(2)) == 2

    def test_evaluate_returns_false(self):
        from finetune_studio.benchmarks import BaseBenchmark
        b = BaseBenchmark()
        assert b.evaluate({}, "anything") is False

    def test_get_expected(self):
        from finetune_studio.benchmarks import BaseBenchmark
        b = BaseBenchmark()
        assert b.get_expected({"answer": 42}) == "42"


class TestMMLUSample:
    def test_has_samples(self):
        from finetune_studio.benchmarks import MMLUSample
        m = MMLUSample()
        assert len(m) > 0
        assert m.category == "knowledge"

    def test_evaluate_correct(self):
        from finetune_studio.benchmarks import MMLUSample
        m = MMLUSample()
        sample = m.samples[0]
        assert m.evaluate(sample, "Paris") is True

    def test_evaluate_incorrect(self):
        from finetune_studio.benchmarks import MMLUSample
        m = MMLUSample()
        sample = m.samples[0]
        assert m.evaluate(sample, "Berlin") is False

    def test_format_prompt(self):
        from finetune_studio.benchmarks import MMLUSample
        m = MMLUSample()
        prompt = m.format_prompt(m.samples[0])
        assert "capital of France" in prompt


class TestHellaSwagSample:
    def test_evaluate_correct(self):
        from finetune_studio.benchmarks import HellaSwagSample
        h = HellaSwagSample()
        assert h.evaluate(h.samples[0], "B") is True
        assert h.evaluate(h.samples[0], "B) The person continues") is True

    def test_evaluate_incorrect(self):
        from finetune_studio.benchmarks import HellaSwagSample
        h = HellaSwagSample()
        assert h.evaluate(h.samples[0], "A") is False

    def test_category(self):
        from finetune_studio.benchmarks import HellaSwagSample
        assert HellaSwagSample().category == "reasoning"


class TestIFEvalSample:
    def test_evaluate_three_items(self):
        from finetune_studio.benchmarks import IFEvalSample
        iv = IFEvalSample()
        sample = iv.samples[0]
        assert iv.evaluate(sample, "Python\nJavaScript\nRust") is True

    def test_evaluate_wrong_count(self):
        from finetune_studio.benchmarks import IFEvalSample
        iv = IFEvalSample()
        sample = iv.samples[0]
        assert iv.evaluate(sample, "Python\nJavaScript") is False


class TestToolBenchSample:
    def test_evaluate_keyword_match(self):
        from finetune_studio.benchmarks import ToolBenchSample
        tb = ToolBenchSample()
        sample = tb.samples[0]
        assert tb.evaluate(sample, "I would use web_search for this") is True

    def test_evaluate_no_match(self):
        from finetune_studio.benchmarks import ToolBenchSample
        tb = ToolBenchSample()
        sample = tb.samples[0]
        assert tb.evaluate(sample, "I have no idea") is False


class TestGSM8KSample:
    def test_evaluate_correct(self):
        from finetune_studio.benchmarks import GSM8KSample
        g = GSM8KSample()
        assert g.evaluate(g.samples[0], "The answer is 8") is True

    def test_evaluate_incorrect(self):
        from finetune_studio.benchmarks import GSM8KSample
        g = GSM8KSample()
        assert g.evaluate(g.samples[0], "The answer is 10") is False


class TestTruthfulQASample:
    def test_evaluate_truthful(self):
        from finetune_studio.benchmarks import TruthfulQASample
        tq = TruthfulQASample()
        sample = tq.samples[2]
        assert tq.evaluate(sample, "No, the earth is round") is True

    def test_evaluate_hallucination(self):
        from finetune_studio.benchmarks import TruthfulQASample
        tq = TruthfulQASample()
        sample = tq.samples[2]
        assert tq.evaluate(sample, "Yes, the earth is flat") is False


class TestPersonaTest:
    def test_evaluate_substantive(self):
        from finetune_studio.benchmarks import PersonaTest
        pt = PersonaTest()
        assert pt.evaluate(pt.samples[0], "My name is Chris AI") is True

    def test_evaluate_too_short(self):
        from finetune_studio.benchmarks import PersonaTest
        pt = PersonaTest()
        assert pt.evaluate(pt.samples[0], "No") is False


class TestBenchmarkSuite:
    def test_list_benchmarks(self):
        from finetune_studio.benchmarks import BenchmarkSuite
        suite = BenchmarkSuite()
        names = suite.list_benchmarks()
        assert isinstance(names, list)
        assert len(names) > 0
        assert "name" in names[0]
        assert "description" in names[0]

    def test_run_benchmark_unknown_raises(self):
        from finetune_studio.benchmarks import BenchmarkSuite
        suite = BenchmarkSuite()
        with pytest.raises(ValueError, match="Unknown benchmark"):
            suite.run_benchmark(MagicMock(), "nonexistent_bench")

    def test_run_benchmark_runs_samples(self, mock_engine):
        from finetune_studio.benchmarks import BenchmarkSuite
        mock_engine.generate.return_value = {"response": "Paris", "tokens": 10, "time_ms": 50}
        suite = BenchmarkSuite()
        result = suite.run_benchmark(mock_engine, "mmlu_sample", num_samples=2)
        assert result["benchmark"] == "mmlu_sample"
        assert result["total"] == 2
        assert "accuracy" in result
        assert "results" in result
        assert mock_engine.generate.call_count == 2

    def test_run_benchmark_with_error(self, mock_engine):
        """run_benchmark handles errors gracefully."""
        from finetune_studio.benchmarks import BenchmarkSuite
        # Make generate fail for ALL calls
        mock_engine.generate.side_effect = RuntimeError("Model crashed")
        suite = BenchmarkSuite()
        result = suite.run_benchmark(mock_engine, "mmlu_sample", num_samples=2)
        assert result["total"] == 2
        assert result["passed"] == 0

    def test_run_all(self, mock_engine):
        from finetune_studio.benchmarks import BenchmarkSuite
        mock_engine.generate.return_value = {"response": "test", "tokens": 5, "time_ms": 20}
        suite = BenchmarkSuite()
        result = suite.run_all(mock_engine, num_samples=1)
        assert "benchmarks" in result
        assert "aggregate" in result
        assert result["aggregate"]["num_benchmarks"] > 0


class TestModelComparator:
    def test_init(self):
        from finetune_studio.benchmarks.comparison import ModelComparator
        mc = ModelComparator()
        assert mc.engines == {}

    def test_run_comparison_dict_tests(self, mock_engine):
        from finetune_studio.benchmarks.comparison import ModelComparator
        mc = ModelComparator()
        mc.engines["test_model"] = mock_engine
        test_suite = [
            {
                "name": "test1",
                "messages": [{"role": "user", "content": "Hello"}],
                "expected": {"keywords": ["hi"], "forbidden": []},
            }
        ]
        result = mc.run_comparison(test_suite)
        assert "results" in result
        assert "summary" in result
        assert "test_model" in result["summary"]

    def test_run_comparison_object_tests(self, mock_engine):
        from finetune_studio.benchmarks.comparison import ModelComparator
        mc = ModelComparator()
        mc.engines["m"] = mock_engine
        test = MagicMock()
        test.name = "t1"
        test.messages = [{"role": "user", "content": "Hi"}]
        test.expected_keywords = ["hello"]
        test.forbidden_keywords = []
        result = mc.run_comparison([test])
        assert len(result["results"]) == 1

    def test_score_response_with_keywords(self):
        from finetune_studio.benchmarks.comparison import ModelComparator
        mc = ModelComparator()
        score = mc._score_response("Hello world", {"keywords": ["hello"], "forbidden": []})
        assert score["correct"] is True

    def test_score_response_no_check(self):
        from finetune_studio.benchmarks.comparison import ModelComparator
        mc = ModelComparator()
        score = mc._score_response("Anything", {})
        assert score["correct"] is True
        assert score["method"] == "no_check"

    def test_cleanup(self, mock_engine):
        from finetune_studio.benchmarks.comparison import ModelComparator
        mc = ModelComparator()
        mc.engines["m"] = mock_engine
        mc.cleanup()
        assert mc.engines == {}
        mock_engine.unload.assert_called()

    def test_comparator_singleton(self):
        from finetune_studio.benchmarks.comparison import comparator, ModelComparator
        assert isinstance(comparator, ModelComparator)


# ══════════════════════════════════════════════════════════════
# tool_calling.py tests
# ══════════════════════════════════════════════════════════════

class TestToolCallEvaluator:
    def test_parse_tool_call_xml_format(self):
        from finetune_studio.benchmarks.tool_calling import ToolCallEvaluator
        evaluator = ToolCallEvaluator()
        response = '<tool_call>\n{"name": "web_search", "arguments": {"query": "python"}}\n</tool_call>'
        tc = evaluator.parse_tool_call(response)
        assert tc is not None
        assert tc.name == "web_search"
        assert tc.arguments["query"] == "python"

    def test_parse_tool_call_json_format(self):
        from finetune_studio.benchmarks.tool_calling import ToolCallEvaluator
        evaluator = ToolCallEvaluator()
        response = '{"tool": "calculator", "arguments": {"expression": "2+2"}}'
        tc = evaluator.parse_tool_call(response)
        assert tc is not None
        assert tc.name == "calculator"

    def test_parse_tool_call_no_match(self):
        from finetune_studio.benchmarks.tool_calling import ToolCallEvaluator
        evaluator = ToolCallEvaluator()
        assert evaluator.parse_tool_call("Just a normal response") is None

    def test_evaluate_tool_call_correct(self):
        from finetune_studio.benchmarks.tool_calling import ToolCallEvaluator, ToolCall, AgenticTest
        evaluator = ToolCallEvaluator()
        tc = ToolCall(name="web_search", arguments={"query": "python"})
        test = AgenticTest(
            name="test", description="d", system_prompt="s", user_message="m",
            expected_tools=["web_search"],
        )
        result = evaluator.evaluate_tool_call(tc, test)
        assert result["correct"] is True

    def test_evaluate_tool_call_wrong_tool(self):
        from finetune_studio.benchmarks.tool_calling import ToolCallEvaluator, ToolCall, AgenticTest
        evaluator = ToolCallEvaluator()
        tc = ToolCall(name="calculator", arguments={})
        test = AgenticTest(
            name="test", description="d", system_prompt="s", user_message="m",
            expected_tools=["web_search"],
        )
        result = evaluator.evaluate_tool_call(tc, test)
        assert result["correct"] is False

    def test_evaluate_tool_call_forbidden(self):
        from finetune_studio.benchmarks.tool_calling import ToolCallEvaluator, ToolCall, AgenticTest
        evaluator = ToolCallEvaluator()
        tc = ToolCall(name="calculator", arguments={})
        test = AgenticTest(
            name="test", description="d", system_prompt="s", user_message="m",
            expected_tools=[], forbidden_tools=["calculator"],
        )
        result = evaluator.evaluate_tool_call(tc, test)
        assert result["correct"] is False

    def test_evaluate_no_tool_expected_none_called(self):
        from finetune_studio.benchmarks.tool_calling import ToolCallEvaluator, AgenticTest
        evaluator = ToolCallEvaluator()
        test = AgenticTest(
            name="test", description="d", system_prompt="s", user_message="m",
            expected_tools=[],
        )
        result = evaluator.evaluate_tool_call(None, test)
        assert result["correct"] is True

    def test_evaluate_tool_expected_none_called(self):
        from finetune_studio.benchmarks.tool_calling import ToolCallEvaluator, AgenticTest
        evaluator = ToolCallEvaluator()
        test = AgenticTest(
            name="test", description="d", system_prompt="s", user_message="m",
            expected_tools=["web_search"],
        )
        result = evaluator.evaluate_tool_call(None, test)
        assert result["correct"] is False

    def test_tool_call_tests_exist(self):
        from finetune_studio.benchmarks.tool_calling import TOOL_CALL_TESTS
        assert len(TOOL_CALL_TESTS) > 0
        assert all(hasattr(t, "name") for t in TOOL_CALL_TESTS)

    def test_tool_definitions_exist(self):
        from finetune_studio.benchmarks.tool_calling import TOOL_DEFINITIONS
        assert len(TOOL_DEFINITIONS) > 0
        assert all("function" in t for t in TOOL_DEFINITIONS)

    def test_get_tool_system_prompt(self):
        from finetune_studio.benchmarks.tool_calling import get_tool_system_prompt
        prompt = get_tool_system_prompt()
        assert "web_search" in prompt
        assert "calculator" in prompt


# ══════════════════════════════════════════════════════════════
# real_benchmarks.py tests (mocked)
# ══════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestRealBenchmarkSuite:
    def test_list_available(self):
        from finetune_studio.benchmarks.real_benchmarks import RealBenchmarkSuite
        rbs = RealBenchmarkSuite(cache_dir="/tmp/test_cache")
        available = rbs.list_available()
        assert "mmlu" in available
        assert "hellaswag" in available
        assert "arc_challenge" in available
        assert "truthfulqa" in available
        assert "gsm8k" in available
        assert "winogrande" in available

    def test_evaluate_mmlu_mocked(self):
        from finetune_studio.benchmarks.real_benchmarks import RealBenchmarkSuite
        mock_ds = [
            {"question": "What is 2+2?", "answer": 1, "subject": "math",
             "choices": ["3", "4", "5", "6"]},
            {"question": "Capital of France?", "answer": 2, "subject": "geography",
             "choices": ["Berlin", "Paris", "London", "Rome"]},
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = lambda self: iter(mock_ds)
        mock_dataset.__len__ = lambda self: len(mock_ds)
        mock_dataset.filter.return_value = mock_dataset
        mock_dataset.select.return_value = mock_dataset

        mock_engine = MagicMock()
        mock_engine.generate.return_value = {"response": "B", "tokens": 5, "time_ms": 20}

        # Patch load_dataset in sys.modules since it's imported lazily
        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_dataset
        import sys
        with patch.dict(sys.modules, {"datasets": mock_datasets}):
            rbs = RealBenchmarkSuite(cache_dir="/tmp/test_cache")
            result = rbs.evaluate_mmlu(mock_engine, num_samples=2)
            assert result["benchmark"] == "mmlu"
            assert result["total"] == 2
            assert "accuracy" in result

    def test_evaluate_hellaswag_mocked(self):
        from finetune_studio.benchmarks.real_benchmarks import RealBenchmarkSuite
        mock_ds = [
            {"activity_label": "eating", "ctx": "A person is eating.", "label": "0",
             "endings": ["continues", "teleports", "flies", "disappears"]},
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = lambda self: iter(mock_ds)
        mock_dataset.__len__ = lambda self: len(mock_ds)
        mock_dataset.select.return_value = mock_dataset

        mock_engine = MagicMock()
        mock_engine.generate.return_value = {"response": "A", "tokens": 5, "time_ms": 20}

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_dataset
        import sys
        with patch.dict(sys.modules, {"datasets": mock_datasets}):
            rbs = RealBenchmarkSuite(cache_dir="/tmp/test_cache")
            result = rbs.evaluate_hellaswag(mock_engine, num_samples=1)
            assert result["benchmark"] == "hellaswag"
            assert result["total"] == 1

    def test_evaluate_arc_mocked(self):
        from finetune_studio.benchmarks.real_benchmarks import RealBenchmarkSuite
        mock_ds = [
            {"question": "What is H2O?", "choices": {"label": ["A", "B"], "text": ["Water", "Fire"]},
             "answerKey": "A"},
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = lambda self: iter(mock_ds)
        mock_dataset.__len__ = lambda self: len(mock_ds)
        mock_dataset.select.return_value = mock_dataset

        mock_engine = MagicMock()
        mock_engine.generate.return_value = {"response": "A", "tokens": 5, "time_ms": 20}

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_dataset
        import sys
        with patch.dict(sys.modules, {"datasets": mock_datasets}):
            rbs = RealBenchmarkSuite(cache_dir="/tmp/test_cache")
            result = rbs.evaluate_arc(mock_engine, num_samples=1)
            assert result["benchmark"] == "arc_challenge"

    def test_evaluate_gsm8k_mocked(self):
        from finetune_studio.benchmarks.real_benchmarks import RealBenchmarkSuite
        mock_ds = [
            {"question": "2+2=?", "answer": "4\n#### 4"},
        ]
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = lambda self: iter(mock_ds)
        mock_dataset.__len__ = lambda self: len(mock_ds)
        mock_dataset.select.return_value = mock_dataset

        mock_engine = MagicMock()
        mock_engine.generate.return_value = {"response": "4\n#### 4", "tokens": 10, "time_ms": 30}

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_dataset
        import sys
        with patch.dict(sys.modules, {"datasets": mock_datasets}):
            rbs = RealBenchmarkSuite(cache_dir="/tmp/test_cache")
            result = rbs.evaluate_gsm8k(mock_engine, num_samples=1)
            assert result["benchmark"] == "gsm8k"

    def test_run_all_mocked(self):
        from finetune_studio.benchmarks.real_benchmarks import RealBenchmarkSuite
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = lambda self: iter([])
        mock_dataset.__len__ = lambda self: 0
        mock_dataset.filter.return_value = mock_dataset
        mock_dataset.select.return_value = mock_dataset

        mock_engine = MagicMock()

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_dataset
        import sys
        with patch.dict(sys.modules, {"datasets": mock_datasets}):
            rbs = RealBenchmarkSuite(cache_dir="/tmp/test_cache")
            result = rbs.run_all(mock_engine, num_samples=0, benchmarks=["mmlu", "gsm8k"])
            assert "benchmarks" in result
            assert "summary" in result
