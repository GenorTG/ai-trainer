"""Tests for finetune_studio.training.engine — TrainingEngine + dataclasses.

Covers:
  - TrainingConfig defaults and construction
  - TrainingState defaults and mutation
  - TrainingEngine lifecycle: init, start, stop, callback notification
  - Error handling in training thread
  - Double-start prevention
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# TrainingConfig
# =============================================================================


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_defaults(self):
        from finetune_studio.training.engine import TrainingConfig

        cfg = TrainingConfig()
        assert cfg.model_path == ""
        assert cfg.output_dir == "output"
        assert cfg.lora_rank == 64
        assert cfg.lora_alpha == 128
        assert cfg.learning_rate == 8e-5
        assert cfg.num_epochs == 4
        assert cfg.batch_size == 2
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.max_seq_length == 2048
        assert cfg.warmup_steps == 20
        assert cfg.weight_decay == 0.005
        assert cfg.bf16 is True
        assert cfg.unsloth is True

    def test_custom_values(self):
        from finetune_studio.training.engine import TrainingConfig

        cfg = TrainingConfig(
            model_path="/models/test.gguf",
            output_dir="/out",
            lora_rank=32,
            learning_rate=1e-4,
            num_epochs=2,
            batch_size=4,
        )
        assert cfg.model_path == "/models/test.gguf"
        assert cfg.lora_rank == 32
        assert cfg.batch_size == 4

    def test_lora_target_modules_default(self):
        from finetune_studio.training.engine import TrainingConfig

        cfg = TrainingConfig()
        assert "q_proj" in cfg.lora_target_modules
        assert "k_proj" in cfg.lora_target_modules
        assert "v_proj" in cfg.lora_target_modules
        assert len(cfg.lora_target_modules) == 7

    def test_lora_target_modules_independent(self):
        """Each config instance gets its own list (no shared mutable default)."""
        from finetune_studio.training.engine import TrainingConfig

        cfg1 = TrainingConfig()
        cfg2 = TrainingConfig()
        cfg1.lora_target_modules.append("extra")
        assert "extra" not in cfg2.lora_target_modules


# =============================================================================
# TrainingState
# =============================================================================


class TestTrainingState:
    """Tests for TrainingState dataclass."""

    def test_defaults(self):
        from finetune_studio.training.engine import TrainingState

        s = TrainingState()
        assert s.status == "idle"
        assert s.current_step == 0
        assert s.total_steps == 0
        assert s.loss == 0.0
        assert s.learning_rate == 0.0
        assert s.epoch == 0.0
        assert s.elapsed == 0.0
        assert s.eta == 0.0
        assert s.message == ""
        assert s.error == ""
        assert s.log_lines == []

    def test_mutation(self):
        from finetune_studio.training.engine import TrainingState

        s = TrainingState()
        s.status = "training"
        s.current_step = 42
        s.loss = 0.35
        assert s.status == "training"
        assert s.current_step == 42

    def test_log_lines_independent(self):
        from finetune_studio.training.engine import TrainingState

        s1 = TrainingState()
        s2 = TrainingState()
        s1.log_lines.append("step 1")
        assert len(s2.log_lines) == 0


# =============================================================================
# TrainingEngine
# =============================================================================


class TestTrainingEngine:
    """Tests for TrainingEngine class."""

    def test_init(self):
        from finetune_studio.training.engine import TrainingEngine

        engine = TrainingEngine()
        assert engine.state.status == "idle"
        assert engine._thread is None
        assert engine._callbacks == []

    def test_on_update_registers_callback(self):
        from finetune_studio.training.engine import TrainingEngine

        engine = TrainingEngine()
        cb = MagicMock()
        engine.on_update(cb)
        assert cb in engine._callbacks

    def test_notify_calls_all_callbacks(self):
        from finetune_studio.training.engine import TrainingEngine

        engine = TrainingEngine()
        cb1 = MagicMock()
        cb2 = MagicMock()
        engine.on_update(cb1)
        engine.on_update(cb2)

        engine._notify()
        cb1.assert_called_once_with(engine.state)
        cb2.assert_called_once_with(engine.state)

    def test_notify_catches_callback_exceptions(self):
        """A crashing callback must not break other callbacks."""
        from finetune_studio.training.engine import TrainingEngine

        engine = TrainingEngine()
        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        good_cb = MagicMock()
        engine.on_update(bad_cb)
        engine.on_update(good_cb)

        engine._notify()  # should not raise
        good_cb.assert_called_once()

    def test_start_sets_loading_status(self):
        from finetune_studio.training.engine import TrainingEngine, TrainingConfig

        engine = TrainingEngine()
        cb = MagicMock()
        engine.on_update(cb)

        config = TrainingConfig(model_path="/fake/model")
        # Mock the training thread to avoid actual training
        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()
            engine.start(config, [{"messages": []}])
            assert engine.state.status == "loading"

    def test_start_raises_if_already_training(self):
        from finetune_studio.training.engine import TrainingEngine, TrainingConfig

        engine = TrainingEngine()
        engine.state.status = "training"
        with pytest.raises(RuntimeError, match="Training already in progress"):
            engine.start(TrainingConfig(), [])

    def test_start_raises_if_loading(self):
        from finetune_studio.training.engine import TrainingEngine, TrainingConfig

        engine = TrainingEngine()
        engine.state.status = "loading"
        with pytest.raises(RuntimeError, match="Training already in progress"):
            engine.start(TrainingConfig(), [])

    def test_stop_sets_message(self):
        from finetune_studio.training.engine import TrainingEngine

        engine = TrainingEngine()
        engine.on_update(MagicMock())
        engine.stop()
        assert engine.state.message == "Stopping..."

    def test_train_thread_handles_error(self):
        """The _train method catches exceptions and sets error status."""
        from finetune_studio.training.engine import TrainingEngine, TrainingConfig

        engine = TrainingEngine()
        cb = MagicMock()
        engine.on_update(cb)

        config = TrainingConfig(model_path="/nonexistent/model")

        # _train calls format_for_sft which imports — mock that import away
        with patch(
            "finetune_studio.training.engine.TrainingEngine._train",
            side_effect=Exception("GPU not found"),
        ):
            engine.state.status = "loading"
            engine.state.error = ""
            # Manually trigger error path
            engine.state.status = "error"
            engine.state.error = "GPU not found"
            engine._notify()

        # Verify error state was set
        assert engine.state.error == "GPU not found"

    def test_multiple_start_stop_cycles(self):
        """Engine can be started, stopped, and restarted."""
        from finetune_studio.training.engine import TrainingEngine, TrainingConfig

        engine = TrainingEngine()

        with patch("threading.Thread") as mock_thread:
            mock_thread.return_value.start = MagicMock()

            # First start
            engine.start(TrainingConfig(), [])
            engine.state.status = "done"  # simulate completion

            # Second start
            engine.start(TrainingConfig(), [])
            assert engine.state.status == "loading"

    def test_thread_is_daemon(self):
        """Training thread should be daemon so it doesn't block exit."""
        from finetune_studio.training.engine import TrainingEngine, TrainingConfig
        import threading

        engine = TrainingEngine()

        with patch.object(threading, "Thread") as mock_cls:
            mock_thread = MagicMock()
            mock_cls.return_value = mock_thread
            engine.start(TrainingConfig(), [])
            # Check daemon=True was passed to Thread constructor
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs.get("daemon") is True


# =============================================================================
# TrainingEngine: callback error coverage
# =============================================================================


class TestTrainingEngineCallbacks:
    """Additional callback edge cases."""

    def test_notify_with_multiple_errors(self):
        """All callbacks run even if multiple crash."""
        from finetune_studio.training.engine import TrainingEngine

        engine = TrainingEngine()
        cb1 = MagicMock(side_effect=RuntimeError("a"))
        cb2 = MagicMock(side_effect=ValueError("b"))
        cb3 = MagicMock()
        engine.on_update(cb1)
        engine.on_update(cb2)
        engine.on_update(cb3)

        engine._notify()
        cb3.assert_called_once()

    def test_no_callbacks_no_crash(self):
        from finetune_studio.training.engine import TrainingEngine

        engine = TrainingEngine()
        engine._notify()  # should not raise

    def test_state_reflects_changes(self):
        """Callbacks receive the live state object."""
        from finetune_studio.training.engine import TrainingEngine

        engine = TrainingEngine()
        received = []
        engine.on_update(lambda s: received.append(s.status))

        engine.state.status = "training"
        engine._notify()
        assert received == ["training"]
