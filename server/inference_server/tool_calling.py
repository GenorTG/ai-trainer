"""Tool-call data structures and evaluation.

WHAT THIS FILE DOES
==================
Defines the data structures for tool calls (ToolCall, ToolDefinition)
and provides evaluation logic — given a model's response, does it
correctly call the expected tool with the right arguments?

KEY CONCEPTS
============
- Tool definition: describes a tool the model can call (name, description,
  parameters). Follows OpenAI's function calling format.
- Tool call: a model's request to invoke a tool (tool name + arguments).
- Evaluation: did the model call the right tool? With the right arguments?
  We use simple string matching and exact-match comparison.
- Test cases: predefined scenarios like "ask the model to search for
  X, then save a note" — used to measure tool-call accuracy.
"""

import json
import re
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw: str = ""


@dataclass
class AgenticTest:
    name: str
    description: str
    system_prompt: str
    user_message: str
    expected_tools: list
    expected_args: dict = field(default_factory=dict)
    forbidden_tools: list = field(default_factory=list)
    category: str = "tool_calling"


class ToolCallEvaluator:
    def parse_tool_call(self, response: str) -> ToolCall | None:
        text = response.strip()
        text = text.replace("<|channel>thought", "").replace("</start_of_turn>", "")
        text = text.replace("<start_of_turn>model\n", "").replace("<|end|>", "")
        text = text.replace("end_of_turn>", "").replace("<end_of_turn>", "")
        text = text.strip()

        # Format 1: tool_calls array
        match = re.search(r'"tool_calls":\s*\[(.*?)\]', text, re.DOTALL)
        if match:
            try:
                items = json.loads("[" + match.group(1) + "]")
                if items:
                    item = items[0]
                    name = item.get("name", "")
                    args = item.get("args", item.get("arguments", {}))
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except Exception:  # noqa: BLE001
                            args = {}
                    return ToolCall(name=name, arguments=args or {}, raw=text)
            except Exception:  # noqa: BLE001, S110
                pass

        # Format 2: Single JSON object
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                name = data.get("name", data.get("tool", ""))
                args = data.get("arguments", data.get("args", {}))
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except Exception:  # noqa: BLE001
                        args = {}
                if name:
                    return ToolCall(name=name, arguments=args or {}, raw=text)
        except Exception:  # noqa: BLE001, S110
            pass

        # Format 3: <tool_call> wrapper
        match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                name = data.get("name", data.get("tool", ""))
                args = data.get("arguments", data.get("args", {}))
                return ToolCall(name=name, arguments=args or {}, raw=text)
            except Exception:  # noqa: BLE001, S110
                pass

        # Format 4: Search for JSON with name/tool key
        match = re.search(r'\{[^{}]*"(?:name|tool)"[^{}]*\}', text)
        if match:
            try:
                data = json.loads(match.group(0))
                name = data.get("name", data.get("tool", ""))
                args = data.get("arguments", data.get("args", {}))
                if name:
                    return ToolCall(name=name, arguments=args or {}, raw=text)
            except Exception:  # noqa: BLE001, S110
                pass

        return None

    def evaluate_tool_call(self, tool_call: ToolCall | None, test) -> dict:
        if tool_call is None:
            if not test.expected_tools:
                return {"correct": True, "tool_called": None, "reason": "No tool needed"}
            return {"correct": False, "tool_called": None, "reason": "No tool call detected"}

        correct_tool = tool_call.name in test.expected_tools
        forbidden = tool_call.name in test.forbidden_tools

        if not test.expected_tools:
            return {"correct": False, "tool_called": tool_call.name, "reason": "Tool called but none expected"}

        return {
            "correct": correct_tool and not forbidden,
            "tool_called": tool_call.name,
            "arguments": tool_call.arguments,
        }


TOOL_CALL_TESTS = [
    AgenticTest("web_search_python", "Should call web_search", "You are a helpful assistant.", "Search for Python programming", ["web_search"], {"query": ["python"]}),
    AgenticTest("calculator_math", "Should call calculator", "You are a helpful assistant.", "What is 15 * 37 + 42?", ["calculator"], {"expression": ["15", "37"]}),
    AgenticTest("file_read", "Should call file_read", "You are a helpful assistant.", "Read /home/user/document.txt", ["file_read"], {"path": ["document.txt", "/home/user"]}),
    AgenticTest("note_save", "Should call note_save", "You are a helpful assistant.", "Save: Meeting at 3pm", ["note_save"], {"content": ["meeting", "3pm"]}),
    AgenticTest("rag_search", "Should call rag_search", "You have access to a knowledge base.", "What projects completed?", ["rag_search"], {"query": ["project"]}),
    AgenticTest("weather", "Should call weather_check", "You are a helpful assistant.", "Weather in Warsaw?", ["weather_check"], {"city": ["Warsaw"]}),
    AgenticTest("no_tool_simple", "Should NOT use tool for 2+2", "You are a helpful assistant.", "What is 2 + 2?", [], {}),
    AgenticTest("no_tool_greeting", "Should NOT use tool for greeting", "You are a helpful assistant.", "Hello! How are you?", [], {}),
    AgenticTest("multi_step", "Should search then save", "You are a helpful assistant.", "Search climate change and save a note", ["web_search", "note_save"]),
]

TOOL_PROMPT_SUFFIX = "\n\nYou have access to these tools. When you need to use one, respond with ONLY a JSON object:\n{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"value\"}}\n\nAvailable tools:\n- web_search: Search the web\n- calculator: Calculate math\n- file_read: Read a file\n- note_save: Save a note\n- rag_search: Search knowledge base\n- weather_check: Check weather\n\nIf no tool is needed, respond normally.\n"
