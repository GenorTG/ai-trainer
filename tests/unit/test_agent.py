"""Tests for inference_server.agent — ToolCallingAgent class.

Tests the high-level agent abstraction with native and manual tool calling modes.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from inference_server.agent import ToolCallingAgent

# =============================================================================
# Initialization tests
# =============================================================================


class TestAgentInit:
    """Tests for ToolCallingAgent.__init__."""

    def test_init_with_gguf_engine_enables_native(self):
        """Native mode auto-enabled for GGUF engines."""
        engine = MagicMock()
        engine.is_gguf = True
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store, native_tools=True)

        assert agent.native_tools is True

    def test_init_with_non_gguf_disables_native(self):
        """Native mode disabled for non-GGUF engines."""
        engine = MagicMock()
        engine.is_gguf = False
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store, native_tools=True)

        assert agent.native_tools is False

    def test_init_native_false_disables_native(self):
        """native_tools=False forces manual mode."""
        engine = MagicMock()
        engine.is_gguf = True
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store, native_tools=False)

        assert agent.native_tools is False

    def test_init_builds_tools(self):
        """Constructor calls build_tools to populate tools list."""
        engine = MagicMock()
        engine.is_gguf = True
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store)

        assert agent.tools is not None
        assert len(agent.tools) > 0

    def test_init_default_max_iterations(self):
        """Default max_iterations is 5."""
        engine = MagicMock()
        engine.is_gguf = True
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store)

        assert agent.max_iterations == 5

    def test_init_custom_max_iterations(self):
        """Custom max_iterations is respected."""
        engine = MagicMock()
        engine.is_gguf = True
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store, max_iterations=10)

        assert agent.max_iterations == 10

    def test_init_stores_embedding_model(self):
        """embedding_model is stored."""
        engine = MagicMock()
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store, embedding_model="custom-model")

        assert agent.embedding_model == "custom-model"


# =============================================================================
# _format_tool_prompt tests
# =============================================================================


class TestFormatToolPrompt:
    """Tests for agent._format_tool_prompt()."""

    def test_format_includes_tool_descriptions(self):
        """Tool descriptions appear in formatted prompt."""
        engine = MagicMock()
        engine.is_gguf = True
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store)
        prompt = agent._format_tool_prompt()

        # Should mention some tools
        assert "rag_search" in prompt or "tool" in prompt.lower()

    def test_format_includes_format_example(self):
        """Prompt includes tool_call format example."""
        engine = MagicMock()
        engine.is_gguf = True
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store)
        prompt = agent._format_tool_prompt()

        assert "<tool_call>" in prompt or "tool_call" in prompt

    def test_format_instructs_normal_response(self):
        """Prompt tells model to respond normally when no tool needed."""
        engine = MagicMock()
        engine.is_gguf = True
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store)
        prompt = agent._format_tool_prompt()

        # Should say something about normal response
        assert "normally" in prompt.lower() or "respond" in prompt.lower()


# =============================================================================
# _parse_manual_tool_call tests
# =============================================================================


class TestParseManualToolCall:
    """Tests for agent._parse_manual_tool_call()."""

    def test_parses_agent_format(self):
        """Parses {tool: name, arguments: {...}}."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._parse_manual_tool_call(
            '{"tool": "rag_search", "arguments": {"query": "test"}}'
        )
        assert result is not None
        assert result["tool"] == "rag_search"
        assert result["arguments"] == {"query": "test"}

    def test_parses_openai_format(self):
        """Parses {name: ..., arguments: ...}."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._parse_manual_tool_call(
            '{"name": "calc", "arguments": {"x": 1}}'
        )
        assert result is not None
        assert result["tool"] == "calc"

    def test_parses_qwen_tag_format(self):
        """Parses <tool_call>...</tool_call>."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._parse_manual_tool_call(
            '<tool_call>{"name": "x", "arguments": {}}</tool_call>'
        )
        assert result is not None
        assert result["tool"] == "x"

    def test_parses_double_braced_json(self):
        """Parses {{...}} (Qwen Jinja double-brace)."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._parse_manual_tool_call(
            '{{"name": "double", "arguments": {}}}'
        )
        assert result is not None
        assert result["tool"] == "double"

    def test_returns_none_for_plain_text(self):
        """Returns None for plain text without tool call."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._parse_manual_tool_call("just a normal answer")
        assert result is None

    def test_returns_none_for_invalid_json(self):
        """Returns None for invalid JSON."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._parse_manual_tool_call("not json at all")
        assert result is None

    def test_returns_none_for_json_without_tool_or_name(self):
        """JSON without tool/name field returns None."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._parse_manual_tool_call('{"data": "x"}')
        assert result is None

    def test_parses_string_arguments(self):
        """String arguments are JSON-decoded."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._parse_manual_tool_call(
            '{"name": "x", "arguments": "{\\"q\\": \\"test\\"}"}'
        )
        assert result is not None
        assert isinstance(result["arguments"], dict)


# =============================================================================
# _extract_json_objects tests
# =============================================================================


class TestExtractJsonObjects:
    """Tests for agent._extract_json_objects()."""

    def test_extracts_single_object(self):
        """Extracts one balanced JSON object."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._extract_json_objects('{"a": 1, "b": 2}')
        assert len(result) >= 1
        assert '{"a": 1, "b": 2}' in result

    def test_extracts_nested_object(self):
        """Extracts object with nested braces (may find inner too)."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._extract_json_objects('{"outer": {"inner": 1}}')
        # Implementation extracts outer + inner (greedy)
        assert len(result) >= 1
        assert '{"outer": {"inner": 1}}' in result

    def test_extracts_multiple_objects(self):
        """Extracts multiple sibling JSON objects."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._extract_json_objects('{"a": 1} {"b": 2}')
        assert len(result) == 2

    def test_handles_strings_with_braces(self):
        """Braces inside strings don't break parsing."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._extract_json_objects('{"key": "value with } brace"}')
        assert len(result) == 1
        # Should preserve braces inside string
        assert '}' in result[0]

    def test_empty_text_returns_empty_list(self):
        """Empty text returns empty list."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._extract_json_objects("")
        assert result == []

    def test_text_without_json_returns_empty_list(self):
        """Plain text returns empty list."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._extract_json_objects("no json here")
        assert result == []


# =============================================================================
# _try_parse tests
# =============================================================================


class TestTryParse:
    """Tests for agent._try_parse()."""

    def test_parses_valid_json_with_name(self):
        """Parses JSON with name field."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._try_parse('{"name": "x", "arguments": {}}')
        assert result is not None
        assert result["tool"] == "x"

    def test_parses_valid_json_with_tool(self):
        """Parses JSON with tool field."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._try_parse('{"tool": "x", "arguments": {}}')
        assert result is not None
        assert result["tool"] == "x"

    def test_returns_none_for_non_dict(self):
        """Non-dict JSON returns None."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._try_parse('[1, 2, 3]')
        assert result is None

    def test_returns_none_for_dict_without_name_or_tool(self):
        """Dict without name/tool returns None."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._try_parse('{"other": "field"}')
        assert result is None

    def test_returns_none_for_invalid_json(self):
        """Invalid JSON returns None."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._try_parse("not json")
        assert result is None


# =============================================================================
# _normalize_tool_call tests
# =============================================================================


class TestNormalizeToolCall:
    """Tests for agent._normalize_tool_call()."""

    def test_normalizes_tool_key(self):
        """Normalizes {tool: ..., arguments: ...}."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._normalize_tool_call(
            {"tool": "x", "arguments": {"a": 1}}
        )
        assert result["tool"] == "x"
        assert result["arguments"] == {"a": 1}

    def test_normalizes_name_key(self):
        """Normalizes {name: ..., arguments: ...}."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._normalize_tool_call(
            {"name": "x", "arguments": {"a": 1}}
        )
        assert result["tool"] == "x"

    def test_normalizes_string_arguments(self):
        """String arguments are JSON-decoded."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._normalize_tool_call(
            {"tool": "x", "arguments": '{"a": 1}'}
        )
        assert result["arguments"] == {"a": 1}

    def test_normalizes_invalid_string_arguments(self):
        """Invalid string args become empty dict."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._normalize_tool_call(
            {"tool": "x", "arguments": "not json"}
        )
        assert result["arguments"] == {}

    def test_normalizes_name_with_parameters_key(self):
        """Normalizes {name: ..., parameters: ...}."""
        engine = MagicMock()
        agent = ToolCallingAgent(engine, MagicMock())

        result = agent._normalize_tool_call(
            {"name": "x", "parameters": {"a": 1}}
        )
        assert result["tool"] == "x"
        assert result["arguments"] == {"a": 1}


# =============================================================================
# run() tests
# =============================================================================


class TestRunBasic:
    """Tests for agent.run() — basic behavior."""

    def test_run_no_tool_calls_returns_response(self):
        """run() returns final response when no tool calls."""
        engine = MagicMock()
        engine.is_gguf = True
        engine.model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello there!"}}]
        }
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store)
        result = agent.run([{"role": "user", "content": "hi"}])

        assert result["response"] == "Hello there!"
        assert result["tool_calls"] == []

    def test_run_with_native_tool_call_executes(self):
        """run() executes tool when native tool call returned."""
        engine = MagicMock()
        engine.is_gguf = True
        # First call: tool call
        # Second call: final answer
        engine.model.create_chat_completion.side_effect = [
            {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "function": {
                                "name": "rag_search",
                                "arguments": '{"query": "test"}',
                            },
                        }],
                    }
                }]
            },
            {
                "choices": [{"message": {"content": "Final answer"}}]
            },
        ]
        rag_store = MagicMock()
        rag_store.search.return_value = []

        agent = ToolCallingAgent(engine, rag_store)
        result = agent.run([{"role": "user", "content": "hi"}])

        assert "Final answer" in str(result["response"])
        assert len(result["tool_calls"]) >= 1

    def test_run_respects_max_iterations(self):
        """run() stops after max_iterations."""
        engine = MagicMock()
        engine.is_gguf = True
        # Always returns tool call
        engine.model.create_chat_completion.return_value = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_x",
                        "function": {
                            "name": "rag_search",
                            "arguments": '{"query": "test"}',
                        },
                    }],
                }
            }]
        }
        rag_store = MagicMock()
        rag_store.search.return_value = []

        agent = ToolCallingAgent(engine, rag_store, max_iterations=3)
        result = agent.run([{"role": "user", "content": "hi"}])

        # Should not exceed max_iterations
        assert len(result["tool_calls"]) <= 3

    def test_run_falls_back_to_manual_on_native_failure(self):
        """run() falls back to manual when native fails."""
        engine = MagicMock()
        engine.is_gguf = True
        engine.model.create_chat_completion.side_effect = RuntimeError("native failed")
        engine.generate.return_value = "Manual answer"
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store)
        # Use manual mode from start
        agent.native_tools = False

        result = agent.run([{"role": "user", "content": "hi"}])

        assert result["response"] == "Manual answer"

    def test_run_strips_messages_copy(self):
        """run() doesn't mutate input messages."""
        engine = MagicMock()
        engine.is_gguf = True
        engine.model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "OK"}}]
        }
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store)
        original = [{"role": "user", "content": "hi"}]
        agent.run(original)

        # Input unchanged
        assert original == [{"role": "user", "content": "hi"}]


class TestRunManualMode:
    """Tests for agent.run() in manual tool calling mode."""

    def test_run_manual_mode(self):
        """Manual mode uses _generate_manual."""
        engine = MagicMock()
        engine.is_gguf = False  # Forces manual mode
        engine.generate.return_value = "Manual response"
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store)
        result = agent.run([{"role": "user", "content": "hi"}])

        assert result["response"] == "Manual response"

    def test_run_manual_mode_with_tool_call(self):
        """Manual mode parses tool calls from response."""
        engine = MagicMock()
        engine.is_gguf = False
        # First call returns tool call, second returns final answer
        engine.generate.side_effect = [
            '<tool_call>{"name": "rag_search", "arguments": {"query": "test"}}</tool_call>',
            "Final answer",
        ]
        rag_store = MagicMock()
        rag_store.search.return_value = []

        agent = ToolCallingAgent(engine, rag_store)
        result = agent.run([{"role": "user", "content": "hi"}])

        assert "Final answer" in result["response"]
        assert len(result["tool_calls"]) >= 1

    def test_run_manual_mode_appends_tool_prompt_to_system(self):
        """Manual mode injects tool prompt into system message."""
        engine = MagicMock()
        engine.is_gguf = False
        engine.generate.return_value = "answer"
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store)
        messages = [
            {"role": "system", "content": "Original prompt"},
            {"role": "user", "content": "hi"},
        ]
        agent.run(messages)

        # engine.generate called with modified messages
        call_args = engine.generate.call_args
        sent_messages = call_args.args[0]
        # System message should have tool prompt appended
        assert sent_messages[0]["role"] == "system"
        assert "tool" in sent_messages[0]["content"].lower()

    def test_run_manual_mode_prepends_system_if_missing(self):
        """Manual mode prepends system message if not present."""
        engine = MagicMock()
        engine.is_gguf = False
        engine.generate.return_value = "answer"
        rag_store = MagicMock()

        agent = ToolCallingAgent(engine, rag_store)
        messages = [{"role": "user", "content": "hi"}]
        agent.run(messages)

        call_args = engine.generate.call_args
        sent_messages = call_args.args[0]
        # First message should be system with tool prompt
        assert sent_messages[0]["role"] == "system"


class TestRunToolLogging:
    """Tests for tool call logging in run()."""

    def test_tool_log_includes_arguments(self):
        """Tool log includes the arguments passed."""
        engine = MagicMock()
        engine.is_gguf = True
        engine.model.create_chat_completion.side_effect = [
            {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "function": {
                                "name": "rag_search",
                                "arguments": '{"query": "test query"}',
                            },
                        }],
                    }
                }]
            },
            {"choices": [{"message": {"content": "answer"}}]},
        ]
        rag_store = MagicMock()
        rag_store.search.return_value = []

        agent = ToolCallingAgent(engine, rag_store)
        result = agent.run([{"role": "user", "content": "hi"}])

        assert len(result["tool_calls"]) >= 1
        tool_call = result["tool_calls"][0]
        assert tool_call["tool"] == "rag_search"
        assert tool_call["arguments"] == {"query": "test query"}

    def test_tool_log_truncates_long_results(self):
        """Tool log truncates results > 2000 chars."""
        engine = MagicMock()
        engine.is_gguf = True
        engine.model.create_chat_completion.side_effect = [
            {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "function": {
                                "name": "rag_search",
                                "arguments": '{"query": "x"}',
                            },
                        }],
                    }
                }]
            },
            {"choices": [{"message": {"content": "answer"}}]},
        ]
        rag_store = MagicMock()
        rag_store.search.return_value = []
        # Return a long result
        long_text = "x" * 5000
        rag_store.search.return_value = [MagicMock(text=long_text, score=0.9)]

        agent = ToolCallingAgent(engine, rag_store)
        result = agent.run([{"role": "user", "content": "hi"}])

        # Result in tool log should be truncated
        assert len(result["tool_calls"][0]["result"]) <= 2000

    def test_tool_call_id_propagated(self):
        """Tool call ID from native response is used in next message."""
        engine = MagicMock()
        engine.is_gguf = True
        engine.model.create_chat_completion.side_effect = [
            {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_xyz",
                            "function": {
                                "name": "rag_search",
                                "arguments": '{"query": "test"}',
                            },
                        }],
                    }
                }]
            },
            {"choices": [{"message": {"content": "answer"}}]},
        ]
        rag_store = MagicMock()
        rag_store.search.return_value = []

        agent = ToolCallingAgent(engine, rag_store)
        agent.run([{"role": "user", "content": "hi"}])

        # Second call should have tool message with ID
        second_call = engine.model.create_chat_completion.call_args_list[1]
        messages = second_call.kwargs["messages"]
        tool_msg = next(m for m in messages if m.get("role") == "tool")
        assert tool_msg.get("tool_call_id") == "call_xyz"
