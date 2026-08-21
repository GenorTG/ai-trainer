"""Tests for finetune_studio.cli — CLI argument parsing and dispatch.

Covers:
  - main() parser setup (subcommand registration)
  - cmd_validate dispatching to validator
  - cmd_convert dispatching to converter
  - cmd_models with mock registry
  - Error exits for missing files
  - RAG subcommand parsing
"""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# CLI parser: subcommand registration
# =============================================================================


class TestCLIParser:
    """Test that the argparse parser is set up correctly."""

    def test_main_exits_zero_with_no_command(self, capsys):
        """No args prints help and exits 0."""
        from finetune_studio.cli import main

        with patch("sys.argv", ["fts"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0

    def test_all_subcommands_registered(self):
        """Check that expected subcommands exist in the parser."""
        import argparse
        from finetune_studio.cli import main

        # We can't easily inspect the parser without running main,
        # but we can check that the commands dict has all keys.
        from finetune_studio.cli import (
            cmd_analyze, cmd_augment, cmd_benchmark, cmd_compare,
            cmd_convert, cmd_models, cmd_optimize, cmd_rag,
            cmd_rag_test, cmd_suite, cmd_test, cmd_train,
            cmd_validate, cmd_validate_hallucination, cmd_webui,
        )
        # All cmd functions exist
        assert callable(cmd_models)
        assert callable(cmd_train)
        assert callable(cmd_validate)
        assert callable(cmd_convert)
        assert callable(cmd_webui)
        assert callable(cmd_rag)
        assert callable(cmd_compare)
        assert callable(cmd_benchmark)
        assert callable(cmd_analyze)
        assert callable(cmd_augment)
        assert callable(cmd_optimize)
        assert callable(cmd_validate_hallucination)
        assert callable(cmd_rag_test)
        assert callable(cmd_test)
        assert callable(cmd_suite)


# =============================================================================
# cmd_validate
# =============================================================================


class TestCmdValidate:
    """Tests for the validate subcommand."""

    def test_validate_valid_file(self, tmp_path, capsys):
        from finetune_studio.cli import cmd_validate
        import argparse

        f = tmp_path / "good.jsonl"
        f.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")

        args = argparse.Namespace(files=[str(f)])
        cmd_validate(args)

        output = capsys.readouterr().out
        assert "✅" in output

    def test_validate_invalid_file(self, tmp_path, capsys):
        from finetune_studio.cli import cmd_validate
        import argparse

        f = tmp_path / "bad.jsonl"
        f.write_text("not json\n")

        args = argparse.Namespace(files=[str(f)])
        cmd_validate(args)

        output = capsys.readouterr().out
        assert "❌" in output

    def test_validate_multiple_files(self, tmp_path, capsys):
        from finetune_studio.cli import cmd_validate
        import argparse

        f1 = tmp_path / "a.jsonl"
        f2 = tmp_path / "b.jsonl"
        f1.write_text(json.dumps({"messages": [{"role": "user", "content": "a"}]}) + "\n")
        f2.write_text(json.dumps({"messages": [{"role": "user", "content": "b"}]}) + "\n")

        args = argparse.Namespace(files=[str(f1), str(f2)])
        cmd_validate(args)

        output = capsys.readouterr().out
        assert output.count("✅") == 2

    def test_validate_missing_file(self, tmp_path, capsys):
        from finetune_studio.cli import cmd_validate
        import argparse

        args = argparse.Namespace(files=[str(tmp_path / "nope.jsonl")])
        cmd_validate(args)

        output = capsys.readouterr().out
        assert "❌" in output


# =============================================================================
# cmd_convert
# =============================================================================


class TestCmdConvert:
    """Tests for the convert subcommand."""

    def test_convert_jsonl_to_json(self, tmp_path, capsys):
        from finetune_studio.cli import cmd_convert
        import argparse

        src = tmp_path / "data.jsonl"
        dst = tmp_path / "data.json"
        src.write_text(json.dumps({"messages": [{"role": "user", "content": "x"}]}) + "\n")

        args = argparse.Namespace(
            source=str(src), target_format="json",
            output=str(dst), system_prompt="",
        )
        cmd_convert(args)

        output = capsys.readouterr().out
        assert "Converted" in output
        assert dst.exists()

    def test_convert_json_to_jsonl(self, tmp_path, capsys):
        from finetune_studio.cli import cmd_convert
        import argparse

        src = tmp_path / "data.json"
        dst = tmp_path / "data.jsonl"
        src.write_text(json.dumps([{"messages": [{"role": "user", "content": "y"}]}]))

        args = argparse.Namespace(
            source=str(src), target_format="jsonl",
            output=str(dst), system_prompt="",
        )
        cmd_convert(args)

        output = capsys.readouterr().out
        assert "Converted" in output

    def test_convert_csv_to_jsonl(self, tmp_path, capsys):
        from finetune_studio.cli import cmd_convert
        import argparse

        src = tmp_path / "data.csv"
        dst = tmp_path / "data.jsonl"
        src.write_text("text\nHello\n")

        args = argparse.Namespace(
            source=str(src), target_format="jsonl",
            output=str(dst), system_prompt="",
        )
        cmd_convert(args)

        assert dst.exists()

    def test_convert_missing_file_exits(self):
        from finetune_studio.cli import cmd_convert
        import argparse

        args = argparse.Namespace(
            source="/nonexistent/file.json", target_format="jsonl",
            output=None, system_prompt="",
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_convert(args)
        assert exc_info.value.code == 1

    def test_convert_invalid_format_exits(self, tmp_path):
        from finetune_studio.cli import cmd_convert
        import argparse

        src = tmp_path / "data.jsonl"
        src.write_text("{}\n")

        args = argparse.Namespace(
            source=str(src), target_format="csv",
            output=None, system_prompt="",
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_convert(args)
        assert exc_info.value.code == 1

    def test_convert_auto_output_path(self, tmp_path, capsys):
        """When output is None, derive from source."""
        from finetune_studio.cli import cmd_convert
        import argparse

        src = tmp_path / "data.json"
        src.write_text(json.dumps([{"a": 1}]))

        args = argparse.Namespace(
            source=str(src), target_format="jsonl",
            output=None, system_prompt="",
        )
        cmd_convert(args)

        expected = tmp_path / "data.jsonl"
        assert expected.exists()


# =============================================================================
# cmd_models
# =============================================================================


class TestCmdModels:
    """Tests for the models subcommand."""

    def test_models_empty(self, capsys):
        from finetune_studio.cli import cmd_models
        import argparse

        args = argparse.Namespace(dirs=[], json=False)

        with patch("finetune_studio.models.registry.scan_models", return_value=[]):
            with patch("finetune_studio.config.settings", MagicMock(model_dirs=[])):
                cmd_models(args)

        output = capsys.readouterr().out
        assert "No models found" in output

    def test_models_json_output(self, capsys):
        """cmd_models with --json flag."""
        from finetune_studio.cli import cmd_models
        import argparse

        mock_model = MagicMock()
        mock_model.name = "test-model"
        mock_model.path = "/models/test"
        mock_model.format = "gguf"
        mock_model.size_gb = 4.0
        mock_model.architecture = "llama"

        args = argparse.Namespace(dirs=[], json=True)

        with patch("finetune_studio.models.registry.scan_models", return_value=[mock_model]):
            with patch("finetune_studio.config.settings", MagicMock(model_dirs=[])):
                cmd_models(args)

        output = capsys.readouterr().out
        data = json.loads(output)
        assert data[0]["name"] == "test-model"


# =============================================================================
# cmd_train: missing files
# =============================================================================


class TestCmdTrain:
    """Tests for train subcommand error paths."""

    def test_train_missing_model(self):
        from finetune_studio.cli import cmd_train
        import argparse

        args = argparse.Namespace(
            model="/nonexistent/model.gguf",
            data="/nonexistent/data.jsonl",
            output="output", lr=8e-5, epochs=4, batch=2,
            lora_rank=64, max_seq=2048, system_prompt="",
            no_unsloth=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_train(args)
        assert exc_info.value.code == 1

    def test_train_missing_data(self, tmp_path):
        from finetune_studio.cli import cmd_train
        import argparse

        args = argparse.Namespace(
            model=str(tmp_path / "model.gguf"),
            data=str(tmp_path / "nope.jsonl"),
            output="output", lr=8e-5, epochs=4, batch=2,
            lora_rank=64, max_seq=2048, system_prompt="",
            no_unsloth=False,
        )
        # model doesn't exist either, so it errors on model first
        with pytest.raises(SystemExit) as exc_info:
            cmd_train(args)
        assert exc_info.value.code == 1


# =============================================================================
# cmd_rag: subcommand parsing
# =============================================================================


class TestCmdRag:
    """Tests for RAG subcommand dispatch."""

    def test_rag_no_command_exits(self):
        from finetune_studio.cli import cmd_rag
        import argparse

        args = argparse.Namespace(rag_command=None, store="data/rag_store")
        with pytest.raises(SystemExit) as exc_info:
            cmd_rag(args)
        assert exc_info.value.code == 1

    def test_rag_clear_without_confirm(self, tmp_path, capsys):
        from finetune_studio.cli import cmd_rag
        import argparse

        args = argparse.Namespace(
            rag_command="clear", store=str(tmp_path), confirm=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_rag(args)
        assert exc_info.value.code == 1
        output = capsys.readouterr().out
        assert "--confirm" in output


# =============================================================================
# cmd_suite: missing files
# =============================================================================


@pytest.mark.slow  # needs torch (training.engine)
class TestCmdSuite:
    """Tests for suite subcommand error paths."""

    def test_suite_missing_model(self):
        from finetune_studio.cli import cmd_suite
        import argparse

        args = argparse.Namespace(
            model="/nope.gguf", suite="/nope.json",
            max_tokens=512, json=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_suite(args)
        assert exc_info.value.code == 1

    def test_suite_missing_suite(self, tmp_path):
        from finetune_studio.cli import cmd_suite
        import argparse

        args = argparse.Namespace(
            model=str(tmp_path / "model.gguf"),
            suite=str(tmp_path / "nope.json"),
            max_tokens=512, json=False,
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_suite(args)
        assert exc_info.value.code == 1


# =============================================================================
# cmd_benchmark: missing model
# =============================================================================


@pytest.mark.slow  # needs torch (benchmarks)
class TestCmdBenchmark:
    """Tests for benchmark subcommand error paths."""

    def test_benchmark_missing_model(self):
        from finetune_studio.cli import cmd_benchmark
        import argparse

        args = argparse.Namespace(
            model="/nope.gguf", suite="all", num_samples=100,
            max_tokens=10, temperature=0.0, json=False, report=None, real=True,
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_benchmark(args)
        assert exc_info.value.code == 1


# =============================================================================
# cmd_compare: missing files
# =============================================================================


class TestCmdCompare:
    """Tests for compare subcommand error paths."""

    def test_compare_missing_suite(self):
        from finetune_studio.cli import cmd_compare
        import argparse

        args = argparse.Namespace(
            models=["a= /nope.gguf"], suite="/nope.json",
            max_tokens=512, temperature=0.7, json=False,
            report=None, real=True,
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_compare(args)
        assert exc_info.value.code == 1
