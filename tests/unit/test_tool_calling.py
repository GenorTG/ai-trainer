"""Tests for inference_server/tool_calling.py."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestToolCall:
    def test_dataclass_fields(self):
        from inference_server.tool_calling import ToolCall
        tc = ToolCall(name="search", arguments={"q": "hello"}, raw="raw text")
        assert tc.name == "search"
        assert tc.arguments == {"q": "hello"}
        assert tc.raw == "raw text"

    def test_defaults(self):
        from inference_server.tool_calling import ToolCall
        tc = ToolCall(name="a", arguments={})
        assert tc.raw == ""


@pytest.mark.unit
class TestAgenticTest:
    def test_dataclass_fields(self):
        from inference_server.tool_calling import AgenticTest
        at = AgenticTest(
            name="test1",
            description="A test",
            system_prompt="sys",
            user_message="msg",
            expected_tools=["web_search"],
            expected_args={"q": ["python"]},
            forbidden_tools=["calculator"],
            category="tool_calling",
        )
        assert at.name == "test1"
        assert at.expected_tools == ["web_search"]
        assert at.forbidden_tools == ["calculator"]

    def test_defaults(self):
        from inference_server.tool_calling import AgenticTest
        at = AgenticTest(
            name="t",
            description="d",
            system_prompt="s",
            user_message="m",
            expected_tools=[],
        )
        assert at.expected_args == {}
        assert at.forbidden_tools == []
        assert at.category == "tool_calling"


@pytest.mark.unit
class TestToolCallEvaluator:
    @pytest.fixture
    def evaluator(self):
        from inference_server.tool_calling import ToolCallEvaluator
        return ToolCallEvaluator()

    # ── parse_tool_call formats ──

    def test_parse_single_json(self, evaluator):
        from inference_server.tool_calling import ToolCall
        result = evaluator.parse_tool_call('{"name": "search", "arguments": {"q": "test"}}')
        assert isinstance(result, ToolCall)
        assert result.name == "search"
        assert result.arguments == {"q": "test"}

    def test_parse_tool_call_wrapper(self, evaluator):
        from inference_server.tool_calling import ToolCall
        text = '<tool_call>\n{"name": "calc", "arguments": {"expr": "2+2"}}\n</tool_call>'
        result = evaluator.parse_tool_call(text)
        assert isinstance(result, ToolCall)
        assert result.name == "calc"
        assert result.arguments["expr"] == "2+2"

    def test_parse_tool_calls_array(self, evaluator):
        from inference_server.tool_calling import ToolCall
        text = '{"tool_calls": [{"name": "search", "args": {"q": "hello"}}]}'
        result = evaluator.parse_tool_call(text)
        assert isinstance(result, ToolCall)
        assert result.name == "search"

    def test_parse_json_with_tool_key(self, evaluator):
        from inference_server.tool_calling import ToolCall
        text = '{"tool": "calculator", "arguments": {"expr": "1+1"}}'
        result = evaluator.parse_tool_call(text)
        assert isinstance(result, ToolCall)
        assert result.name == "calculator"

    def test_parse_no_tool_call(self, evaluator):
        result = evaluator.parse_tool_call("Just a normal response with no tool calls.")
        assert result is None

    def test_parse_empty_string(self, evaluator):
        result = evaluator.parse_tool_call("")
        assert result is None

    def test_parse_strips_thinking_tokens(self, evaluator):
        from inference_server.tool_calling import ToolCall
        # After the token-stripping, the remainder is valid single-JSON
        # that parse_tool_call can decode via format 2.
        text = '<|channel>thought</start_of_turn>{"name": "search", "arguments": {}}'
        result = evaluator.parse_tool_call(text)
        assert isinstance(result, ToolCall)
        assert result.name == "search"

    def test_parse_args_as_string(self, evaluator):
        """When arguments are a JSON string instead of dict."""
        from inference_server.tool_calling import ToolCall
        text = '{"name": "calc", "arguments": "{\\"expr\\": \\"1+1\\"}"}'
        result = evaluator.parse_tool_call(text)
        # Should try to parse the string
        assert isinstance(result, ToolCall)

    def test_parse_malformed_json_in_wrapper(self, evaluator):
        text = '<tool_call>not json</tool_call>'
        result = evaluator.parse_tool_call(text)
        assert result is None

    # ── evaluate_tool_call ──

    def test_evaluate_correct_tool(self, evaluator):
        from inference_server.tool_calling import ToolCall, AgenticTest
        tc = ToolCall(name="web_search", arguments={"q": "test"})
        test = AgenticTest(
            name="t", description="d", system_prompt="s", user_message="m",
            expected_tools=["web_search"],
        )
        result = evaluator.evaluate_tool_call(tc, test)
        assert result["correct"] is True

    def test_evaluate_wrong_tool(self, evaluator):
        from inference_server.tool_calling import ToolCall, AgenticTest
        tc = ToolCall(name="calculator", arguments={})
        test = AgenticTest(
            name="t", description="d", system_prompt="s", user_message="m",
            expected_tools=["web_search"],
        )
        result = evaluator.evaluate_tool_call(tc, test)
        assert result["correct"] is False

    def test_evaluate_forbidden_tool(self, evaluator):
        from inference_server.tool_calling import ToolCall, AgenticTest
        tc = ToolCall(name="calculator", arguments={})
        test = AgenticTest(
            name="t", description="d", system_prompt="s", user_message="m",
            expected_tools=["calculator", "web_search"],
            forbidden_tools=["calculator"],
        )
        result = evaluator.evaluate_tool_call(tc, test)
        assert result["correct"] is False

    def test_evaluate_no_tool_expected_but_called(self, evaluator):
        from inference_server.tool_calling import ToolCall, AgenticTest
        tc = ToolCall(name="web_search", arguments={})
        test = AgenticTest(
            name="t", description="d", system_prompt="s", user_message="m",
            expected_tools=[],
        )
        result = evaluator.evaluate_tool_call(tc, test)
        assert result["correct"] is False

    def test_evaluate_no_tool_detected_expected_none(self, evaluator):
        from inference_server.tool_calling import AgenticTest
        test = AgenticTest(
            name="t", description="d", system_prompt="s", user_message="m",
            expected_tools=[],
        )
        result = evaluator.evaluate_tool_call(None, test)
        assert result["correct"] is True

    def test_evaluate_no_tool_detected_but_expected(self, evaluator):
        from inference_server.tool_calling import AgenticTest
        test = AgenticTest(
            name="t", description="d", system_prompt="s", user_message="m",
            expected_tools=["web_search"],
        )
        result = evaluator.evaluate_tool_call(None, test)
        assert result["correct"] is False
        assert "No tool call detected" in result["reason"]


@pytest.mark.unit
class TestToolCallTests:
    def test_tool_call_tests_defined(self):
        from inference_server.tool_calling import TOOL_CALL_TESTS
        assert len(TOOL_CALL_TESTS) >= 8
        names = {t.name for t in TOOL_CALL_TESTS}
        assert "web_search_python" in names
        assert "calculator_math" in names
        assert "no_tool_simple" in names

    def test_tool_call_tests_have_required_fields(self):
        from inference_server.tool_calling import TOOL_CALL_TESTS
        for test in TOOL_CALL_TESTS:
            assert test.name
            assert test.description
            assert test.system_prompt
            assert test.user_message
            assert isinstance(test.expected_tools, list)


@pytest.mark.unit
class TestToolPromptSuffix:
    def test_suffix_contains_tools(self):
        from inference_server.tool_calling import TOOL_PROMPT_SUFFIX
        assert "web_search" in TOOL_PROMPT_SUFFIX
        assert "calculator" in TOOL_PROMPT_SUFFIX
        assert "file_read" in TOOL_PROMPT_SUFFIX
        assert "note_save" in TOOL_PROMPT_SUFFIX
        assert "rag_search" in TOOL_PROMPT_SUFFIX
        assert "weather_check" in TOOL_PROMPT_SUFFIX
        assert "JSON" in TOOL_PROMPT_SUFFIX
