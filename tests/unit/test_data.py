"""Tests for finetune_studio.data — converter + validator modules.

Covers:
  - converter: jsonl_to_json, json_to_jsonl, csv_to_jsonl, simple_to_chat
  - validator: validate_file, validate_jsonl
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


# =============================================================================
# converter: jsonl_to_json
# =============================================================================


class TestJsonlToJson:
    """Tests for converter.jsonl_to_json."""

    def test_basic_conversion(self, tmp_path):
        from finetune_studio.data.converter import jsonl_to_json

        src = tmp_path / "data.jsonl"
        dst = tmp_path / "data.json"
        lines = [
            {"messages": [{"role": "user", "content": "hi"}]},
            {"messages": [{"role": "user", "content": "bye"}]},
        ]
        src.write_text("\n".join(json.dumps(l) for l in lines) + "\n")

        jsonl_to_json(str(src), str(dst))

        result = json.loads(dst.read_text())
        assert len(result) == 2
        assert result[0]["messages"][0]["content"] == "hi"

    def test_empty_jsonl(self, tmp_path):
        from finetune_studio.data.converter import jsonl_to_json

        src = tmp_path / "empty.jsonl"
        dst = tmp_path / "empty.json"
        src.write_text("")

        jsonl_to_json(str(src), str(dst))

        result = json.loads(dst.read_text())
        assert result == []

    def test_skips_blank_lines(self, tmp_path):
        from finetune_studio.data.converter import jsonl_to_json

        src = tmp_path / "mixed.jsonl"
        dst = tmp_path / "mixed.json"
        src.write_text('{"a":1}\n\n{"a":2}\n\n')

        jsonl_to_json(str(src), str(dst))

        result = json.loads(dst.read_text())
        assert len(result) == 2

    def test_preserves_unicode(self, tmp_path):
        from finetune_studio.data.converter import jsonl_to_json

        src = tmp_path / "uni.jsonl"
        dst = tmp_path / "uni.json"
        src.write_text(json.dumps({"messages": [{"role": "user", "content": "Привет 你好"}]}) + "\n")

        jsonl_to_json(str(src), str(dst))

        result = json.loads(dst.read_text())
        assert "Привет 你好" in result[0]["messages"][0]["content"]

    def test_roundtrip_jsonl_json_jsonl(self, tmp_path):
        from finetune_studio.data.converter import jsonl_to_json, json_to_jsonl

        original = tmp_path / "orig.jsonl"
        mid = tmp_path / "mid.json"
        final = tmp_path / "final.jsonl"
        data = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
        original.write_text(json.dumps(data[0]) + "\n")

        jsonl_to_json(str(original), str(mid))
        json_to_jsonl(str(mid), str(final))

        lines = [json.loads(l) for l in final.read_text().strip().split("\n") if l.strip()]
        assert lines[0]["messages"][1]["content"] == "a"


# =============================================================================
# converter: json_to_jsonl
# =============================================================================


class TestJsonToJsonl:
    """Tests for converter.json_to_jsonl."""

    def test_basic_conversion(self, tmp_path):
        from finetune_studio.data.converter import json_to_jsonl

        src = tmp_path / "data.json"
        dst = tmp_path / "data.jsonl"
        data = [{"messages": [{"role": "user", "content": "test"}]}]
        src.write_text(json.dumps(data, indent=2))

        json_to_jsonl(str(src), str(dst))

        lines = [json.loads(l) for l in dst.read_text().strip().split("\n")]
        assert len(lines) == 1
        assert lines[0]["messages"][0]["content"] == "test"

    def test_empty_json_array(self, tmp_path):
        from finetune_studio.data.converter import json_to_jsonl

        src = tmp_path / "empty.json"
        dst = tmp_path / "empty.jsonl"
        src.write_text("[]")

        json_to_jsonl(str(src), str(dst))

        assert dst.read_text().strip() == ""

    def test_preserves_special_chars(self, tmp_path):
        from finetune_studio.data.converter import json_to_jsonl

        src = tmp_path / "special.json"
        dst = tmp_path / "special.jsonl"
        src.write_text(json.dumps([{"messages": [{"role": "user", "content": 'Line1\nLine2\tTab'}]}]))

        json_to_jsonl(str(src), str(dst))

        result = json.loads(dst.read_text().strip())
        assert "\n" in result["messages"][0]["content"]


# =============================================================================
# converter: csv_to_jsonl
# =============================================================================


class TestCsvToJsonl:
    """Tests for converter.csv_to_jsonl."""

    def test_basic_csv(self, tmp_path):
        from finetune_studio.data.converter import csv_to_jsonl

        src = tmp_path / "data.csv"
        dst = tmp_path / "data.jsonl"
        src.write_text("text\nHello world\nGoodbye world\n")

        csv_to_jsonl(str(src), str(dst))

        lines = [json.loads(l) for l in dst.read_text().strip().split("\n")]
        assert len(lines) == 2
        assert lines[0]["messages"][0]["content"] == "Hello world"

    def test_with_system_prompt(self, tmp_path):
        from finetune_studio.data.converter import csv_to_jsonl

        src = tmp_path / "data.csv"
        dst = tmp_path / "data.jsonl"
        src.write_text("text\nHi\n")

        csv_to_jsonl(str(src), str(dst), system_prompt="You are helpful.")

        result = json.loads(dst.read_text().strip())
        msgs = result["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."
        assert msgs[1]["role"] == "user"

    def test_custom_text_column(self, tmp_path):
        from finetune_studio.data.converter import csv_to_jsonl

        src = tmp_path / "data.csv"
        dst = tmp_path / "data.jsonl"
        src.write_text("question,answer\nWhat? =answer\n")

        csv_to_jsonl(str(src), str(dst), text_column="question")

        result = json.loads(dst.read_text().strip())
        assert result["messages"][0]["content"] == "What? =answer"

    def test_empty_csv(self, tmp_path):
        from finetune_studio.data.converter import csv_to_jsonl

        src = tmp_path / "empty.csv"
        dst = tmp_path / "empty.jsonl"
        src.write_text("text\n")

        csv_to_jsonl(str(src), str(dst))

        assert dst.read_text().strip() == ""


# =============================================================================
# converter: simple_to_chat
# =============================================================================


class TestSimpleToChat:
    """Tests for converter.simple_to_chat."""

    def test_basic_qa_format(self, tmp_path):
        from finetune_studio.data.converter import simple_to_chat

        src = tmp_path / "chat.txt"
        dst = tmp_path / "chat.jsonl"
        src.write_text("Q: What is 2+2?\nA: 4\n\nQ: What is sky?\nA: Blue\n")

        simple_to_chat(str(src), str(dst))

        lines = [json.loads(l) for l in dst.read_text().strip().split("\n")]
        assert len(lines) == 2
        assert lines[0]["messages"][1]["content"] == "4"

    def test_with_system_prompt(self, tmp_path):
        from finetune_studio.data.converter import simple_to_chat

        src = tmp_path / "chat.txt"
        dst = tmp_path / "chat.jsonl"
        src.write_text("Q: Hello\nA: Hi there\n")

        simple_to_chat(str(src), str(dst), system_prompt="Be friendly.")

        result = json.loads(dst.read_text().strip())
        msgs = result["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_skips_empty_blocks(self, tmp_path):
        from finetune_studio.data.converter import simple_to_chat

        src = tmp_path / "chat.txt"
        dst = tmp_path / "chat.jsonl"
        src.write_text("\n\nQ: Only\nA: One\n\n\n")

        simple_to_chat(str(src), str(dst))

        lines = [json.loads(l) for l in dst.read_text().strip().split("\n")]
        assert len(lines) == 1

    def test_case_insensitive_qa(self, tmp_path):
        from finetune_studio.data.converter import simple_to_chat

        src = tmp_path / "chat.txt"
        dst = tmp_path / "chat.jsonl"
        src.write_text("q: lowercase Q\na: lowercase A\n")

        simple_to_chat(str(src), str(dst))

        lines = [json.loads(l) for l in dst.read_text().strip().split("\n")]
        assert lines[0]["messages"][0]["content"] == "lowercase Q"
        assert lines[0]["messages"][1]["content"] == "lowercase A"


# =============================================================================
# validator: validate_file
# =============================================================================


class TestValidateFile:
    """Tests for validator.validate_file."""

    def test_valid_jsonl(self, tmp_path):
        from finetune_studio.data.validator import validate_file

        f = tmp_path / "good.jsonl"
        data = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
        f.write_text(json.dumps(data) + "\n")

        report = validate_file(str(f))
        assert report["valid"] is True
        assert report["stats"]["rows"] == 1
        assert report["stats"]["messages"] == 2

    def test_file_not_found(self, tmp_path):
        from finetune_studio.data.validator import validate_file

        report = validate_file(str(tmp_path / "nope.jsonl"))
        assert report["valid"] is False
        assert "File not found" in report["errors"][0]

    def test_invalid_json_line(self, tmp_path):
        from finetune_studio.data.validator import validate_file

        f = tmp_path / "bad.jsonl"
        f.write_text("not json\n")

        report = validate_file(str(f))
        assert report["valid"] is False
        assert any("invalid JSON" in e for e in report["errors"])

    def test_missing_role(self, tmp_path):
        from finetune_studio.data.validator import validate_file

        f = tmp_path / "nrole.jsonl"
        f.write_text(json.dumps({"messages": [{"content": "hi"}]}) + "\n")

        report = validate_file(str(f))
        assert report["valid"] is False
        assert any("missing role" in e for e in report["errors"])

    def test_missing_content(self, tmp_path):
        from finetune_studio.data.validator import validate_file

        f = tmp_path / "ncontent.jsonl"
        f.write_text(json.dumps({"messages": [{"role": "user"}]}) + "\n")

        report = validate_file(str(f))
        assert report["valid"] is False
        assert any("missing content" in e for e in report["errors"])

    def test_messages_not_list(self, tmp_path):
        from finetune_studio.data.validator import validate_file

        f = tmp_path / "badlist.jsonl"
        f.write_text(json.dumps({"messages": "not a list"}) + "\n")

        report = validate_file(str(f))
        assert report["valid"] is False
        assert any("must be a list" in e for e in report["errors"])

    def test_no_messages_no_text(self, tmp_path):
        from finetune_studio.data.validator import validate_file

        f = tmp_path / "neither.jsonl"
        f.write_text(json.dumps({"foo": "bar"}) + "\n")

        report = validate_file(str(f))
        assert report["valid"] is True  # not an error, just a warning
        assert any("no messages or text" in w for w in report["warnings"])

    def test_json_file(self, tmp_path):
        from finetune_studio.data.validator import validate_file

        f = tmp_path / "data.json"
        f.write_text(json.dumps([{"messages": [{"role": "user", "content": "x"}]}]))

        report = validate_file(str(f))
        assert report["valid"] is True
        assert report["stats"]["rows"] == 1

    def test_txt_file(self, tmp_path):
        from finetune_studio.data.validator import validate_file

        f = tmp_path / "data.txt"
        f.write_text("line1\nline2\nline3\n")

        report = validate_file(str(f))
        assert report["valid"] is True
        assert report["stats"]["lines"] == 3

    def test_unknown_format(self, tmp_path):
        from finetune_studio.data.validator import validate_file

        f = tmp_path / "data.xyz"
        f.write_text("data")

        report = validate_file(str(f))
        assert report["valid"] is True
        assert any("Unknown format" in w for w in report["warnings"])

    def test_multiple_rows_stats(self, tmp_path):
        from finetune_studio.data.validator import validate_file

        f = tmp_path / "multi.jsonl"
        lines = []
        for i in range(5):
            lines.append(json.dumps({"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}))
        f.write_text("\n".join(lines) + "\n")

        report = validate_file(str(f))
        assert report["valid"] is True
        assert report["stats"]["rows"] == 5
        assert report["stats"]["messages"] == 10


# =============================================================================
# validator: edge cases
# =============================================================================


class TestValidatorEdgeCases:
    """Edge case tests for validator."""

    def test_empty_jsonl_is_valid(self, tmp_path):
        from finetune_studio.data.validator import validate_file

        f = tmp_path / "empty.jsonl"
        f.write_text("")

        report = validate_file(str(f))
        assert report["valid"] is True
        assert report["stats"]["rows"] == 0

    def test_blank_lines_only(self, tmp_path):
        from finetune_studio.data.validator import validate_file

        f = tmp_path / "blanks.jsonl"
        f.write_text("\n\n\n")

        report = validate_file(str(f))
        assert report["valid"] is True

    def test_mixed_valid_invalid(self, tmp_path):
        from finetune_studio.data.validator import validate_file

        f = tmp_path / "mixed.jsonl"
        valid = json.dumps({"messages": [{"role": "user", "content": "ok"}]})
        f.write_text(f"{valid}\nnot json\n{valid}\n")

        report = validate_file(str(f))
        assert report["valid"] is False
        assert report["stats"]["rows"] == 3
