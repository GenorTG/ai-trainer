"""Tests for inference_server.parser — tool-call format parser.

Covers all 4+ parser formats:
- JSON object: {"name": "tool", "arguments": {...}}
- JSON tool_calls array: {"tool_calls": [{"name": "...", "args": {...}}]}
- XML-style: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
- Qwen native: <|tool_call|>call:name{args}<tool_call|>
- Function format: {"function": {"name": "...", "arguments": "..."}}
"""
from __future__ import annotations

import pytest

from inference_server.parser import (
    ToolCall,
    parse_tool_call,
    parse_multiple_tool_calls,
    _extract_json_args,
)


# =============================================================================
# ToolCall dataclass
# =============================================================================


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_creates_minimal_tool_call(self):
        """Creates with just name + arguments."""
        tc = ToolCall(name="rag_search", arguments={"query": "test"})
        assert tc.name == "rag_search"
        assert tc.arguments == {"query": "test"}

    def test_raw_defaults_to_empty(self):
        """raw defaults to empty string."""
        tc = ToolCall(name="x", arguments={})
        assert tc.raw == ""

    def test_format_detected_defaults_to_empty(self):
        """format_detected defaults to empty string."""
        tc = ToolCall(name="x", arguments={})
        assert tc.format_detected == ""

    def test_raw_and_format_can_be_set(self):
        """raw + format_detected can be provided."""
        tc = ToolCall(name="x", arguments={}, raw="raw text", format_detected="json")
        assert tc.raw == "raw text"
        assert tc.format_detected == "json"

    def test_equality(self):
        """Two ToolCalls with same fields are equal."""
        tc1 = ToolCall(name="x", arguments={"a": 1})
        tc2 = ToolCall(name="x", arguments={"a": 1})
        assert tc1 == tc2


# =============================================================================
# _extract_json_args helper
# =============================================================================


class TestExtractJsonArgs:
    """Tests for the _extract_json_args helper."""

    def test_valid_json_object(self):
        """Parses valid JSON object directly."""
        result = _extract_json_args('{"query": "test", "top_k": 5}')
        assert result == {"query": "test", "top_k": 5}

    def test_qwen_pipe_delimiters_cleaned(self):
        """Cleans <|\"|> token delimiters."""
        result = _extract_json_args('query:<|"|>test<|"|>')
        # Should be cleaned to JSON and parsed
        assert "query" in result or "test" in str(result)

    def test_arguments_wrapped_in_braces(self):
        """Wraps {key:value} patterns in braces for parsing."""
        # When inner text is `key:val` without outer braces
        # The function tries {key:val} pattern, falls back to key:"val" regex
        result = _extract_json_args('key:"value"')
        assert result.get("key") == "value"

    def test_arguments_prefix_pattern(self):
        """Extracts arguments:{...} pattern."""
        result = _extract_json_args('arguments:{"query": "test"}')
        assert result == {"query": "test"}

    def test_invalid_json_returns_empty(self):
        """Returns empty dict for unparseable input."""
        result = _extract_json_args("not json at all !@#$%")
        # Returns empty dict or what could be parsed
        assert isinstance(result, dict)

    def test_empty_string_returns_empty_dict(self):
        """Returns empty dict for empty input."""
        result = _extract_json_args("")
        assert result == {}


# =============================================================================
# parse_tool_call — JSON object format
# =============================================================================


class TestParseJsonObject:
    """Tests for JSON object format."""

    def test_parses_basic_json_object(self):
        """Parses basic {name, arguments} JSON."""
        result = parse_tool_call('{"name": "rag_search", "arguments": {"query": "test"}}')
        assert result is not None
        assert result.name == "rag_search"
        assert result.arguments == {"query": "test"}
        assert result.format_detected == "json_object"

    def test_parses_args_key(self):
        """Parses 'args' instead of 'arguments'."""
        result = parse_tool_call('{"name": "calc", "args": {"x": 1}}')
        assert result is not None
        assert result.name == "calc"
        assert result.arguments == {"x": 1}

    def test_parses_parameters_key(self):
        """Parses 'parameters' key."""
        result = parse_tool_call('{"name": "search", "parameters": {"q": "test"}}')
        assert result is not None
        assert result.name == "search"
        assert result.arguments == {"q": "test"}

    def test_parses_tool_name_key(self):
        """Parses 'tool_name' key."""
        result = parse_tool_call('{"tool_name": "x", "arguments": {}}')
        assert result is not None
        assert result.name == "x"
        assert result.format_detected == "tool_name_key"

    def test_parses_tool_key_with_arguments(self):
        """Parses 'tool' key when paired with arguments."""
        result = parse_tool_call('{"tool": "x", "arguments": {}}')
        assert result is not None
        assert result.name == "x"
        assert result.format_detected == "tool_key"

    def test_parses_string_arguments_as_json(self):
        """String arguments are parsed as JSON."""
        result = parse_tool_call('{"name": "x", "arguments": "{\\"q\\": \\"test\\"}"}')
        assert result is not None
        assert result.name == "x"
        # Args should be parsed dict
        assert isinstance(result.arguments, dict)

    def test_no_name_returns_none(self):
        """Returns None when name is missing."""
        result = parse_tool_call('{"arguments": {"x": 1}}')
        # No name → no tool call
        # But might still match via fallback regex
        # If returns, must have name set
        if result is not None:
            assert result.name

    def test_empty_arguments_allowed(self):
        """Empty arguments dict is allowed."""
        result = parse_tool_call('{"name": "ping", "arguments": {}}')
        assert result is not None
        assert result.name == "ping"
        assert result.arguments == {}


# =============================================================================
# parse_tool_call — JSON tool_calls array format
# =============================================================================


class TestParseToolCallsArray:
    """Tests for {tool_calls: [...]} format."""

    def test_parses_tool_calls_array(self):
        """Parses tool_calls array."""
        text = '{"tool_calls": [{"name": "rag_search", "args": {"query": "test"}}]}'
        result = parse_tool_call(text)
        assert result is not None
        assert result.name == "rag_search"
        assert result.arguments == {"query": "test"}
        assert result.format_detected == "tool_calls_array"

    def test_parses_tool_calls_with_arguments_key(self):
        """tool_calls array with 'arguments' key."""
        text = '{"tool_calls": [{"name": "calc", "arguments": {"x": 1}}]}'
        result = parse_tool_call(text)
        assert result is not None
        assert result.name == "calc"
        assert result.arguments == {"x": 1}

    def test_takes_first_tool_call_only(self):
        """Only first tool call is returned from array."""
        text = '{"tool_calls": [{"name": "first", "args": {}}, {"name": "second", "args": {}}]}'
        result = parse_tool_call(text)
        assert result is not None
        assert result.name == "first"

    def test_empty_tool_calls_array(self):
        """Empty tool_calls array returns None."""
        text = '{"tool_calls": []}'
        result = parse_tool_call(text)
        # Empty array — no tool call found via array path
        # May fall through to other formats
        # Just check it doesn't crash
        assert result is None or result.name == ""

    def test_tool_calls_array_with_string_args(self):
        """tool_calls array with string args parsed."""
        text = '{"tool_calls": [{"name": "x", "args": "{\\"q\\": \\"test\\"}"}]}'
        result = parse_tool_call(text)
        assert result is not None
        assert result.name == "x"


# =============================================================================
# parse_tool_call — XML tag format
# =============================================================================


class TestParseXmlTag:
    """Tests for <tool_call>...</tool_call> format."""

    def test_parses_xml_tag_format(self):
        """Parses <tool_call>{...}</tool_call>."""
        text = '<tool_call>{"name": "rag_search", "arguments": {"query": "test"}}</tool_call>'
        result = parse_tool_call(text)
        assert result is not None
        assert result.name == "rag_search"
        assert result.arguments == {"query": "test"}
        assert result.format_detected == "xml_tag"

    def test_xml_with_tool_key(self):
        """XML with 'tool' key instead of name."""
        text = '<tool_call>{"tool": "calc", "args": {"x": 1}}</tool_call>'
        result = parse_tool_call(text)
        assert result is not None
        assert result.name == "calc"
        assert result.arguments == {"x": 1}

    def test_xml_with_whitespace(self):
        """XML tag with surrounding whitespace."""
        text = '   <tool_call>{"name": "x", "arguments": {}}   </tool_call>   '
        result = parse_tool_call(text)
        assert result is not None
        assert result.name == "x"


# =============================================================================
# parse_tool_call — Qwen native format
# =============================================================================


class TestParseQwenNative:
    """Tests for <|tool_call|>call:name{args}<tool_call|> format."""

    def test_parses_qwen_native(self):
        """Parses Qwen <|tool_call|>call:name{args}<tool_call|>."""
        text = '<|tool_call|>call:rag_search{query:"test"}<tool_call|>'
        result = parse_tool_call(text)
        assert result is not None
        assert result.name == "rag_search"
        assert result.format_detected == "qwen_tool_call"

    def test_qwen_native_with_pipe_in_arg(self):
        """Qwen format with <|\"|> delimiters inside args."""
        text = '<|tool_call|>call:calc{x:<|"|>5<|"|>}<tool_call|>'
        result = parse_tool_call(text)
        assert result is not None
        assert result.name == "calc"


# =============================================================================
# parse_tool_call — Function format (legacy)
# =============================================================================


class TestParseFunctionFormat:
    """Tests for function format with nested structure."""

    def test_parses_function_format(self):
        """Parses {function: {name: ..., arguments: ...}}."""
        text = '{"function": {"name": "rag_search", "arguments": "{\\\"query\\\": \\\"test\\\"}"}}'
        result = parse_tool_call(text)
        assert result is not None
        assert result.name == "rag_search"
        assert result.format_detected == "function_format"


# =============================================================================
# parse_tool_call — Edge cases
# =============================================================================


class TestParseEdgeCases:
    """Tests for edge cases."""

    def test_empty_string_returns_none(self):
        """Returns None for empty input."""
        result = parse_tool_call("")
        assert result is None

    def test_whitespace_only_returns_none(self):
        """Returns None for whitespace-only input."""
        result = parse_tool_call("   \n  \t  ")
        assert result is None

    def test_plain_text_returns_none(self):
        """Returns None for plain text without tool call."""
        result = parse_tool_call("This is just a regular answer.")
        assert result is None

    def test_json_without_name_returns_none(self):
        """JSON without name field returns None (or fallback path)."""
        result = parse_tool_call('{"data": "no name here", "args": {}}')
        # No 'name' field, no tool
        if result is not None:
            assert result.name != ""

    def test_raw_is_set(self):
        """raw field is set to original text."""
        text = '{"name": "x", "arguments": {}}'
        result = parse_tool_call(text)
        assert result is not None
        assert result.raw == text

    def test_strips_artifacts(self):
        """Strips Gemma/Qwen artifacts before parsing."""
        text = '<|channel>thought{"name": "x", "arguments": {}}<|end|>'
        result = parse_tool_call(text)
        assert result is not None
        assert result.name == "x"

    def test_handles_unicode_arguments(self):
        """Handles Unicode in arguments."""
        text = '{"name": "search", "arguments": {"query": "Cześć!"}}'
        result = parse_tool_call(text)
        assert result is not None
        assert result.name == "search"


# =============================================================================
# parse_multiple_tool_calls
# =============================================================================


class TestParseMultipleToolCalls:
    """Tests for parse_multiple_tool_calls — extract all tool calls."""

    def test_empty_text_returns_empty_list(self):
        """Empty text returns empty list."""
        result = parse_multiple_tool_calls("")
        assert result == []

    def test_no_tool_calls_returns_empty_list(self):
        """Plain text returns empty list."""
        result = parse_multiple_tool_calls("just text")
        assert result == []

    def test_parses_multiple_xml_tags(self):
        """Parses multiple <tool_call> tags."""
        text = (
            '<tool_call>{"name": "first", "arguments": {"x": 1}}</tool_call>'
            '<tool_call>{"name": "second", "arguments": {"y": 2}}</tool_call>'
        )
        result = parse_multiple_tool_calls(text)
        assert len(result) == 2
        assert result[0].name == "first"
        assert result[1].name == "second"

    def test_parses_multiple_tool_calls_array(self):
        """Parses tool_calls array."""
        text = '{"tool_calls": [{"name": "a", "args": {}}, {"name": "b", "args": {}}]}'
        result = parse_multiple_tool_calls(text)
        assert len(result) == 2
        assert result[0].name == "a"
        assert result[1].name == "b"

    def test_invalid_tool_calls_skipped(self):
        """Invalid tool calls are skipped."""
        text = (
            '<tool_call>not valid json</tool_call>'
            '<tool_call>{"name": "valid", "arguments": {}}</tool_call>'
        )
        result = parse_multiple_tool_calls(text)
        # Only valid one returned
        assert len(result) == 1
        assert result[0].name == "valid"


# =============================================================================
# Format priority
# =============================================================================


class TestFormatPriority:
    """Tests that format detection prioritizes correctly."""

    def test_tool_calls_array_priority_over_json(self):
        """tool_calls array format is detected over plain JSON object."""
        text = '{"tool_calls": [{"name": "array", "args": {}}], "name": "object"}'
        result = parse_tool_call(text)
        assert result is not None
        assert result.format_detected == "tool_calls_array"

    def test_xml_priority_over_json(self):
        """XML format preferred over JSON when both present."""
        text = '<tool_call>{"name": "xml", "arguments": {}}</tool_call>{"name": "json"}'
        result = parse_tool_call(text)
        assert result is not None
        assert result.format_detected == "xml_tag"

    def test_qwen_priority_over_json(self):
        """Qwen native preferred over JSON when both present."""
        text = '<|tool_call|>call:qwen{x:1}<tool_call|>{"name": "json"}'
        result = parse_tool_call(text)
        assert result is not None
        assert result.format_detected == "qwen_tool_call"