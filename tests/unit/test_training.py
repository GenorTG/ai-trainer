"""Tests for finetune_studio.training package.

Covers: engine.py, data_quality.py, data_augmentation.py,
        hallucination_guard.py, knowledge_preservation.py, config_optimizer.py.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

# ══════════════════════════════════════════════════════════════
# engine.py tests
# ══════════════════════════════════════════════════════════════

class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_defaults(self):
        """TrainingConfig has sensible defaults."""
        from finetune_studio.training.engine import TrainingConfig
        tc = TrainingConfig()
        assert tc.model_path == ""
        assert tc.output_dir == "output"
        assert tc.lora_rank == 64
        assert tc.lora_alpha == 128
        assert tc.learning_rate == 8e-5
        assert tc.num_epochs == 4
        assert tc.batch_size == 2
        assert tc.gradient_accumulation_steps == 4
        assert tc.max_seq_length == 2048
        assert tc.bf16 is True
        assert tc.unsloth is True

    def test_custom_values(self):
        """TrainingConfig accepts custom values."""
        from finetune_studio.training.engine import TrainingConfig
        tc = TrainingConfig(lora_rank=32, learning_rate=1e-4, num_epochs=2)
        assert tc.lora_rank == 32
        assert tc.learning_rate == 1e-4
        assert tc.num_epochs == 2

    def test_lora_target_modules(self):
        """Default LoRA target modules include expected layers."""
        from finetune_studio.training.engine import TrainingConfig
        tc = TrainingConfig()
        assert "q_proj" in tc.lora_target_modules
        assert "k_proj" in tc.lora_target_modules
        assert "v_proj" in tc.lora_target_modules


class TestTrainingState:
    """Tests for TrainingState dataclass."""

    def test_defaults(self):
        """TrainingState has idle defaults."""
        from finetune_studio.training.engine import TrainingState
        ts = TrainingState()
        assert ts.status == "idle"
        assert ts.current_step == 0
        assert ts.loss == 0.0
        assert ts.error == ""

    def test_custom_state(self):
        """TrainingState accepts custom values."""
        from finetune_studio.training.engine import TrainingState
        ts = TrainingState(status="training", current_step=100, loss=0.5)
        assert ts.status == "training"
        assert ts.current_step == 100


class TestTrainingEngine:
    """Tests for TrainingEngine class."""

    def test_init(self):
        """TrainingEngine initializes with idle state."""
        from finetune_studio.training.engine import TrainingEngine, TrainingState
        engine = TrainingEngine()
        assert isinstance(engine.state, TrainingState)
        assert engine.state.status == "idle"
        assert engine._callbacks == []

    def test_on_update_registers_callback(self):
        """on_update adds callback to list."""
        from finetune_studio.training.engine import TrainingEngine
        engine = TrainingEngine()
        cb = MagicMock()
        engine.on_update(cb)
        assert cb in engine._callbacks

    def test_notify_calls_callbacks(self):
        """_notify calls all registered callbacks."""
        from finetune_studio.training.engine import TrainingEngine
        engine = TrainingEngine()
        cb1 = MagicMock()
        cb2 = MagicMock()
        engine.on_update(cb1)
        engine.on_update(cb2)
        engine._notify()
        cb1.assert_called_once_with(engine.state)
        cb2.assert_called_once_with(engine.state)

    def test_notify_handles_exception(self):
        """_notify swallows callback exceptions."""
        from finetune_studio.training.engine import TrainingEngine
        engine = TrainingEngine()
        bad_cb = MagicMock(side_effect=RuntimeError("oops"))
        good_cb = MagicMock()
        engine.on_update(bad_cb)
        engine.on_update(good_cb)
        engine._notify()  # Should not raise
        good_cb.assert_called_once()

    def test_start_sets_loading(self):
        """start() sets status to loading and starts thread."""
        from finetune_studio.training.engine import TrainingConfig, TrainingEngine
        engine = TrainingEngine()
        mock_thread = MagicMock()
        with patch("finetune_studio.training.engine.threading.Thread", return_value=mock_thread):
            engine.start(TrainingConfig(), [{"messages": []}])
            assert engine.state.status == "loading"
            mock_thread.start.assert_called_once()

    def test_start_raises_if_already_training(self):
        """start() raises RuntimeError if already training."""
        from finetune_studio.training.engine import TrainingConfig, TrainingEngine, TrainingState
        engine = TrainingEngine()
        engine.state = TrainingState(status="training")
        with pytest.raises(RuntimeError, match="already in progress"):
            engine.start(TrainingConfig(), [])

    def test_stop_sets_event(self):
        """stop() sets the stop event."""
        from finetune_studio.training.engine import TrainingEngine
        engine = TrainingEngine()
        engine.stop()
        assert engine._stop_event.is_set()
        assert engine.state.message == "Stopping..."

    def test_train_error_handling(self):
        """_train catches exceptions and sets error state."""
        from finetune_studio.training.engine import TrainingConfig, TrainingEngine
        engine = TrainingEngine()
        with patch("finetune_studio.training.engine.TrainingEngine._train_unsloth", side_effect=RuntimeError("GPU OOM")):
            engine.config = TrainingConfig(unsloth=True)
            # Manually invoke _train to test error path
            with patch("finetune_studio.training.data.format_for_sft", side_effect=RuntimeError("GPU OOM")):
                engine._train([], "")
                assert engine.state.status == "error"
                assert "GPU OOM" in engine.state.error


# ══════════════════════════════════════════════════════════════
# data_quality.py tests
# ══════════════════════════════════════════════════════════════

class TestDataQualityAnalyzer:
    """Tests for DataQualityAnalyzer class."""

    def _write_jsonl(self, path, data):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    def test_analyze_clean_data(self, tmp_dir):
        """Clean data has no high-severity issues."""
        from finetune_studio.training.data_quality import DataQualityAnalyzer
        p = tmp_dir / "clean.jsonl"
        data = [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
            {"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm fine, thanks!"}]},
        ]
        self._write_jsonl(p, data)
        analyzer = DataQualityAnalyzer()
        result = analyzer.analyze(str(p))
        assert result["total_examples"] == 2
        assert result["severity"] in ("low", "medium")

    def test_analyze_duplicates(self, tmp_dir):
        """Duplicate examples are detected."""
        from finetune_studio.training.data_quality import DataQualityAnalyzer
        p = tmp_dir / "dupes.jsonl"
        item = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
        self._write_jsonl(p, [item, item, item])
        analyzer = DataQualityAnalyzer()
        result = analyzer.analyze(str(p))
        dupe_issues = [i for i in result["issues"] if i["type"] == "duplicates"]
        assert len(dupe_issues) > 0

    def test_analyze_empty_responses(self, tmp_dir):
        """Empty assistant responses are detected."""
        from finetune_studio.training.data_quality import DataQualityAnalyzer
        p = tmp_dir / "empty.jsonl"
        data = [
            {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": ""}]},
        ]
        self._write_jsonl(p, data)
        analyzer = DataQualityAnalyzer()
        result = analyzer.analyze(str(p))
        empty_issues = [i for i in result["issues"] if i["type"] == "empty"]
        assert len(empty_issues) > 0
        assert empty_issues[0]["severity"] == "high"

    def test_analyze_format_errors(self, tmp_dir):
        """Invalid JSON lines are detected."""
        from finetune_studio.training.data_quality import DataQualityAnalyzer
        p = tmp_dir / "bad.jsonl"
        with open(p, "w") as f:
            f.write("not json\n")
            f.write(json.dumps({"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}) + "\n")
        analyzer = DataQualityAnalyzer()
        result = analyzer.analyze(str(p))
        format_issues = [i for i in result["issues"] if i["type"] == "format"]
        assert len(format_issues) > 0

    def test_analyze_language_detection(self, tmp_dir):
        """Polish-heavy dataset triggers language issue."""
        from finetune_studio.training.data_quality import DataQualityAnalyzer
        p = tmp_dir / "pl.jsonl"
        # Use strings with enough Polish diacriticals to exceed the 5% threshold
        data = [
            {"messages": [{"role": "user", "content": "ąćęłńóśźżąćęłńóśźżąćęłńóśźż What is AI?"}, {"role": "assistant", "content": "AI to..."}]},
            {"messages": [{"role": "user", "content": "ąćęłńóśźżąćęłńóśźżąćęłńóśźż How does ML work?"}, {"role": "assistant", "content": "ML to..."}]},
            {"messages": [{"role": "user", "content": "ąćęłńóśźżąćęłńóśźżąćęłńóśźż What is a neural net?"}, {"role": "assistant", "content": "NN to..."}]},
        ]
        self._write_jsonl(p, data)
        analyzer = DataQualityAnalyzer()
        result = analyzer.analyze(str(p))
        # Should detect Polish-heavy dataset
        lang_issues = [i for i in result["issues"] if i["type"] == "language"]
        assert len(lang_issues) > 0

    def test_generate_fixes(self, tmp_dir):
        """generate_fixes produces actionable suggestions."""
        from finetune_studio.training.data_quality import generate_fixes
        analysis = {
            "issues": [
                {"type": "duplicates", "severity": "medium"},
                {"type": "empty", "severity": "high"},
                {"type": "language", "severity": "medium"},
            ]
        }
        fixes = generate_fixes(analysis)
        assert len(fixes) >= 2
        actions = [f["action"] for f in fixes]
        assert "deduplicate" in actions
        assert "remove_empty" in actions

    def test_severity_calculation(self, tmp_dir):
        """_calculate_severity returns highest severity."""
        from finetune_studio.training.data_quality import DataQualityAnalyzer
        analyzer = DataQualityAnalyzer()
        analyzer.issues = [{"severity": "low"}, {"severity": "high"}]
        assert analyzer._calculate_severity() == "high"
        analyzer.issues = [{"severity": "low"}, {"severity": "medium"}]
        assert analyzer._calculate_severity() == "medium"
        analyzer.issues = [{"severity": "low"}]
        assert analyzer._calculate_severity() == "low"


# ══════════════════════════════════════════════════════════════
# data_augmentation.py tests
# ══════════════════════════════════════════════════════════════

class TestDataAugmenter:
    """Tests for DataAugmenter class."""

    def test_generate_knowledge_data(self):
        """generate_knowledge_data returns Q&A pairs."""
        from finetune_studio.training.data_augmentation import DataAugmenter
        aug = DataAugmenter()
        data = aug.generate_knowledge_data(count=10)
        assert len(data) == 10
        assert all("messages" in item for item in data)
        assert all(item["messages"][0]["role"] == "user" for item in data)
        assert all(item["messages"][1]["role"] == "assistant" for item in data)

    def test_generate_refusal_data(self):
        """generate_refusal_data returns refusal examples."""
        from finetune_studio.training.data_augmentation import DataAugmenter
        aug = DataAugmenter()
        data = aug.generate_refusal_data(count=10)
        assert len(data) == 10
        # Refusal responses should indicate limitations or lack of access
        refusal_indicators = ["don't", "can't", "not", "no access", "no single", "depends"]
        for item in data:
            content = item["messages"][1]["content"].lower()
            assert any(w in content for w in refusal_indicators), f"No refusal indicator in: {content[:80]}"

    def test_generate_hallucination_guard(self):
        """generate_hallucination_guard returns guard examples."""
        from finetune_studio.training.data_augmentation import DataAugmenter
        aug = DataAugmenter()
        data = aug.generate_hallucination_guard(count=5)
        assert len(data) == 5

    def test_augment_dataset(self):
        """augment_dataset adds new data based on weaknesses."""
        from finetune_studio.training.data_augmentation import DataAugmenter
        aug = DataAugmenter()
        original = [{"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}]
        augmented = aug.augment_dataset(original, weaknesses=["knowledge"])
        assert len(augmented) > len(original)

    def test_augment_dataset_shuffle(self):
        """augmented dataset is shuffled."""
        from finetune_studio.training.data_augmentation import DataAugmenter
        aug = DataAugmenter()
        original = [{"messages": [{"role": "user", "content": f"Q{i}"}, {"role": "assistant", "content": f"A{i}"}]} for i in range(10)]
        # Run multiple times to verify shuffle exists (non-deterministic)
        results = [aug.augment_dataset(original, weaknesses=["knowledge"]) for _ in range(3)]
        # At least one should differ in order
        assert any(r != results[0] for r in results[1:])

    def test_language_balanced_data(self):
        """generate_language_balanced_data translates Polish to English."""
        from finetune_studio.training.data_augmentation import DataAugmenter
        aug = DataAugmenter()
        pl_data = [
            {"messages": [{"role": "user", "content": "Czym jest AI?"}, {"role": "assistant", "content": "AI is..."}]},
        ]
        result = aug.generate_language_balanced_data(pl_data, target_en=1)
        assert len(result) >= 1
        assert "What is" in result[0]["messages"][0]["content"]

    def test_generators_dict(self):
        """DataAugmenter has all expected generators."""
        from finetune_studio.training.data_augmentation import DataAugmenter
        aug = DataAugmenter()
        assert "knowledge" in aug.generators
        assert "refusal" in aug.generators
        assert "hallucination_guard" in aug.generators


# ══════════════════════════════════════════════════════════════
# hallucination_guard.py tests
# ══════════════════════════════════════════════════════════════

class TestHallucinationGuardrail:
    """Tests for HallucinationGuardrail class."""

    def test_check_clean_response(self):
        """Clean response passes guardrails."""
        from finetune_studio.training.hallucination_guard import HallucinationGuardrail
        guard = HallucinationGuardrail()
        result = guard.check_response("What is Python?", "Python is a programming language.")
        assert result.passed is True

    def test_check_high_confidence(self):
        """High confidence without evidence triggers issue."""
        from finetune_studio.training.hallucination_guard import HallucinationGuardrail
        guard = HallucinationGuardrail()
        result = guard.check_response("Q", "I definitely know this is absolutely correct.")
        assert any("high_confidence" in i for i in result.issues)

    def test_check_fabricated_url(self):
        """Fabricated URL triggers issue."""
        from finetune_studio.training.hallucination_guard import HallucinationGuardrail
        guard = HallucinationGuardrail()
        result = guard.check_response("Q", "Visit https://fake-site.example.com for info.")
        assert any("URL" in i or "url" in i.lower() for i in result.issues)

    def test_check_fabricated_email(self):
        """Fabricated email triggers issue."""
        from finetune_studio.training.hallucination_guard import HallucinationGuardrail
        guard = HallucinationGuardrail()
        result = guard.check_response("Q", "Contact me at fake@example.com for details.")
        assert any("email" in i.lower() for i in result.issues)

    def test_check_short_response(self):
        """Very short response triggers issue."""
        from finetune_studio.training.hallucination_guard import HallucinationGuardrail
        guard = HallucinationGuardrail()
        result = guard.check_response("What is the meaning of life and everything?", "Yes")
        assert any("short" in i.lower() for i in result.issues)

    def test_check_batch(self):
        """check_batch processes multiple QA pairs."""
        from finetune_studio.training.hallucination_guard import HallucinationGuardrail
        guard = HallucinationGuardrail()
        qa_pairs = [
            ("Q1", "Clean answer here"),
            ("Q2", "Another clean answer"),
        ]
        result = guard.check_batch(qa_pairs)
        assert result["total"] == 2
        assert result["passed"] >= 0
        assert "pass_rate" in result

    def test_guardrail_result_dataclass(self):
        """GuardrailResult has expected fields."""
        from finetune_studio.training.hallucination_guard import GuardrailResult
        gr = GuardrailResult(passed=True, issues=["i1"], suggestions=["s1"])
        assert gr.passed is True
        assert len(gr.issues) == 1


class TestTrainingDataValidator:
    """Tests for TrainingDataValidator class."""

    def test_validate_clean_response(self):
        """Clean response has no risks."""
        from finetune_studio.training.hallucination_guard import TrainingDataValidator
        v = TrainingDataValidator()
        risks = v.validate_response("Python is a popular language.")
        assert risks == []

    def test_validate_with_date(self):
        """Specific date triggers risk."""
        from finetune_studio.training.hallucination_guard import TrainingDataValidator
        v = TrainingDataValidator()
        risks = v.validate_response("On 2024-01-15, something happened.")
        assert any(r["type"] == "specific_date" for r in risks)

    def test_validate_with_url(self):
        """URL triggers risk."""
        from finetune_studio.training.hallucination_guard import TrainingDataValidator
        v = TrainingDataValidator()
        risks = v.validate_response("Visit https://example.com for more.")
        assert any(r["type"] == "url" for r in risks)

    def test_validate_with_email(self):
        """Email triggers risk."""
        from finetune_studio.training.hallucination_guard import TrainingDataValidator
        v = TrainingDataValidator()
        risks = v.validate_response("Email user@test.com for info.")
        assert any(r["type"] == "email" for r in risks)

    def test_validate_dataset(self):
        """validate_dataset processes full dataset."""
        from finetune_studio.training.hallucination_guard import TrainingDataValidator
        v = TrainingDataValidator()
        data = [
            {"messages": [{"role": "assistant", "content": "Clean answer"}]},
            {"messages": [{"role": "assistant", "content": "Visit https://fake.com"}]},
        ]
        result = v.validate_dataset(data)
        assert result["total_risks"] >= 1
        assert "risk_types" in result
        assert "recommendation" in result

    def test_recommendation_high_risk(self):
        """High risk ratio returns appropriate recommendation."""
        from finetune_studio.training.hallucination_guard import TrainingDataValidator
        v = TrainingDataValidator()
        rec = v._get_recommendation(10, 20)  # 50% risk
        assert "HIGH" in rec

    def test_recommendation_low_risk(self):
        """Low risk ratio returns low recommendation."""
        from finetune_studio.training.hallucination_guard import TrainingDataValidator
        v = TrainingDataValidator()
        rec = v._get_recommendation(1, 100)  # 1% risk
        assert "LOW" in rec


# ══════════════════════════════════════════════════════════════
# knowledge_preservation.py tests
# ══════════════════════════════════════════════════════════════

class TestKnowledgePreserver:
    """Tests for KnowledgePreserver class."""

    def test_data_mixing(self):
        """data_mixing combines persona and general data."""
        from finetune_studio.training.knowledge_preservation import KnowledgePreserver
        kp = KnowledgePreserver()
        persona = [{"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I'm Chris"}]}] * 10
        general = [{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is..."}]}] * 20
        mixed = kp.data_mixing(persona, general, persona_ratio=0.7)
        assert len(mixed) >= len(persona)
        assert len(mixed) <= len(persona) + len(general)

    def test_replay_buffer(self):
        """replay_buffer extracts general knowledge items."""
        from finetune_studio.training.knowledge_preservation import KnowledgePreserver
        kp = KnowledgePreserver()
        data = [
            {"messages": [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]},
            {"messages": [{"role": "system", "content": "You are Chris"}, {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]},
        ] * 50
        buffer = kp.replay_buffer(data, buffer_size=10)
        assert len(buffer) <= 10
        # All should be non-system items
        for item in buffer:
            assert not any(m.get("role") == "system" for m in item.get("messages", []))

    def test_ewc_hint(self):
        """ewc_hint returns technique description."""
        from finetune_studio.training.knowledge_preservation import KnowledgePreserver
        kp = KnowledgePreserver()
        hint = kp.ewc_hint()
        assert "technique" in hint
        assert "EWC" in hint["technique"]

    def test_progressive_unfreezing_hint(self):
        """progressive_unfreezing_hint returns description."""
        from finetune_studio.training.knowledge_preservation import KnowledgePreserver
        kp = KnowledgePreserver()
        hint = kp.progressive_unfreezing_hint()
        assert "technique" in hint
        assert "Unfreezing" in hint["technique"]

    def test_generate_knowledge_data(self):
        """generate_knowledge_data returns Q&A pairs."""
        from finetune_studio.training.knowledge_preservation import KnowledgePreserver
        kp = KnowledgePreserver()
        data = kp.generate_knowledge_data()
        assert len(data) > 0
        assert all("messages" in item for item in data)

    def test_generate_refusal_data(self):
        """generate_refusal_data returns refusal examples."""
        from finetune_studio.training.knowledge_preservation import KnowledgePreserver
        kp = KnowledgePreserver()
        data = kp.generate_refusal_data()
        assert len(data) > 0

    def test_balance_dataset(self):
        """balance_dataset limits category sizes."""
        from finetune_studio.training.knowledge_preservation import KnowledgePreserver
        kp = KnowledgePreserver()
        data = [
            {"messages": [{"role": "user", "content": f"What is project {i}?"}, {"role": "assistant", "content": f"Project {i}"}]}
            for i in range(20)
        ] + [
            {"messages": [{"role": "user", "content": f"What is AI concept {i}?"}, {"role": "assistant", "content": f"Concept {i}"}]}
            for i in range(5)
        ]
        balanced = kp.balance_dataset(data)
        assert len(balanced) <= len(data)

    def test_techniques_dict(self):
        """KnowledgePreserver has all expected techniques."""
        from finetune_studio.training.knowledge_preservation import KnowledgePreserver
        kp = KnowledgePreserver()
        assert "data_mixing" in kp.techniques
        assert "replay_buffer" in kp.techniques
        assert "elastic_weight_consolidation" in kp.techniques


# ══════════════════════════════════════════════════════════════
# config_optimizer.py tests
# ══════════════════════════════════════════════════════════════

class TestTrainingConfigOptimizer:
    """Tests for TrainingConfigOptimizer class."""

    def test_analyze_small_dataset(self):
        """Small dataset triggers appropriate recommendations."""
        from finetune_studio.training.config_optimizer import TrainingConfigOptimizer
        opt = TrainingConfigOptimizer()
        data = [{"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}] * 100
        recs = opt.analyze_and_recommend(data)
        params = [r.parameter for r in recs]
        assert "learning_rate" in params

    def test_analyze_large_dataset(self):
        """Large dataset triggers different recommendations."""
        from finetune_studio.training.config_optimizer import TrainingConfigOptimizer
        opt = TrainingConfigOptimizer()
        data = [{"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}] * 10000
        recs = opt.analyze_and_recommend(data)
        params = [r.parameter for r in recs]
        assert "learning_rate" in params

    def test_analyze_language_imbalance(self):
        """Language imbalance triggers augmentation recommendation."""
        from finetune_studio.training.config_optimizer import TrainingConfigOptimizer
        opt = TrainingConfigOptimizer()
        data = [{"messages": [{"role": "user", "content": "Czym jest AI?"}, {"role": "assistant", "content": "AI to..."}]}] * 100
        recs = opt.analyze_and_recommend(data)
        params = [r.parameter for r in recs]
        assert "data_augmentation" in params

    def test_analyze_persona_focus(self):
        """High persona ratio triggers data mixing recommendation."""
        from finetune_studio.training.config_optimizer import TrainingConfigOptimizer
        opt = TrainingConfigOptimizer()
        data = [{"messages": [{"role": "user", "content": "Tell me about our company project"}, {"role": "assistant", "content": "Our project..."}]}] * 100
        recs = opt.analyze_and_recommend(data)
        params = [r.parameter for r in recs]
        assert "data_mixing" in params

    def test_generate_report(self):
        """generate_report produces formatted output."""
        from finetune_studio.training.config_optimizer import (
            TrainingConfigOptimizer,
            TrainingRecommendation,
        )
        opt = TrainingConfigOptimizer()
        recs = [
            TrainingRecommendation("lr", "8e-5", "3e-5", "Small dataset", "high"),
            TrainingRecommendation("epochs", "4", "6", "More epochs", "medium"),
        ]
        report = opt.generate_report(recs)
        assert "HIGH Priority:" in report
        assert "MEDIUM Priority:" in report
        assert "lr" in report

    def test_calculate_pl_ratio(self):
        """_calculate_pl_ratio detects Polish text."""
        from finetune_studio.training.config_optimizer import TrainingConfigOptimizer
        opt = TrainingConfigOptimizer()
        # Use strings with enough Polish diacriticals to exceed 5% threshold
        data = [
            {"messages": [{"role": "user", "content": "ąćęłńóśźżąćęłńóśźżąćęłńóśźż What is AI?"}, {"role": "assistant", "content": "AI to..."}]},
            {"messages": [{"role": "user", "content": "ąćęłńóśźżąćęłńóśźżąćęłńóśźż How does ML work?"}, {"role": "assistant", "content": "ML to..."}]},
        ]
        ratio = opt._calculate_pl_ratio(data)
        assert ratio > 0.5

    def test_calculate_persona_ratio(self):
        """_calculate_persona_ratio detects persona keywords."""
        from finetune_studio.training.config_optimizer import TrainingConfigOptimizer
        opt = TrainingConfigOptimizer()
        data = [
            {"messages": [{"role": "user", "content": "Tell me about our company project"}, {"role": "assistant", "content": "..."}]},
            {"messages": [{"role": "user", "content": "What is the team working on?"}, {"role": "assistant", "content": "..."}]},
            {"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "..."}]},
        ]
        ratio = opt._calculate_persona_ratio(data)
        assert ratio > 0.5

    def test_recommendation_dataclass(self):
        """TrainingRecommendation has expected fields."""
        from finetune_studio.training.config_optimizer import TrainingRecommendation
        r = TrainingRecommendation("lr", "8e-5", "3e-5", "reason", "high")
        assert r.parameter == "lr"
        assert r.priority == "high"

    def test_knowledge_preservation_rules(self):
        """knowledge_preservation_rules returns recommendations."""
        from finetune_studio.training.config_optimizer import TrainingConfigOptimizer
        opt = TrainingConfigOptimizer()
        recs = opt._knowledge_preservation_rules({})
        assert len(recs) >= 1
        params = [r.parameter for r in recs]
        assert "general_knowledge_data" in params
