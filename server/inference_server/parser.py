"""Tool-call parser — handles 4+ output formats.

WHAT THIS FILE DOES
==================
Different models output tool calls in different formats. This file
detects which format the model used and extracts the tool name and
arguments. Without this, we couldn't use tool calling consistently
across models.

KEY CONCEPTS
============
- Format 1: JSON — {"name": "tool", "arguments": {...}}
- Format 2: XML-style — <tool_call>{"name": "tool", "args": {...}}</tool_call>
- Format 3: Qwen native — <|tool_call|>call:name{args}<tool_call|>
- Format 4: Gemma native — <|tool_call>call:name{args}<tool_call|>
  (slightly different from Qwen — asymmetric: opening tag has no
  closing pipe, closing tag has no opening pipe)

The regex patterns use pipe-question to handle both the opening (with pipe)
and closing (without pipe) variants of pipes.
"""

import json
import re
from dataclasses import dataclass


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw: str = ""
    format_detected: str = ""


def _extract_json_args(text):
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001, S110
        pass

    cleaned = text.replace('<|"|">', '"')

    try:
        return json.loads(cleaned)
    except Exception:  # noqa: BLE001, S110
        pass

    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            return json.loads(cleaned[1:-1])
        except Exception:  # noqa: BLE001, S110
            pass

    inner = re.search(r'arguments:\{(.+)\}', cleaned)
    if inner:
        try:
            return json.loads("{" + inner.group(1) + "}")
        except Exception:  # noqa: BLE001, S110
            pass

    args = {}
    for pair in re.findall(r'(\w+):\s*"([^"]*)"', cleaned):
        args[pair[0]] = pair[1]
    return args


def parse_tool_call(response):
    text = response.strip()
    for artifact in ["<|channel>thought", "</start_of_turn>", "<start_of_turn>model\n",
                      "<|end|>", "end_of_turn>", "<end_of_turn>", "model\n"]:
        text = text.replace(artifact, "")
    text = text.strip()

    match = re.search(r'"tool_calls"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if match:
        try:
            items = json.loads("[" + match.group(1) + "]")
            if items and isinstance(items, list):
                item = items[0]
                name = item.get("name", "")
                args = item.get("args", item.get("arguments", item.get("parameters", {})))
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except Exception:  # noqa: BLE001
                        args = {}
                if name:
                    return ToolCall(name=name, arguments=args or {}, raw=text, format_detected="tool_calls_array")
        except Exception:  # noqa: BLE001, S110
            pass

    match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        try:
            data = json.loads(content)
            name = data.get("name", data.get("tool", ""))
            args = data.get("arguments", data.get("args", {}))
            if isinstance(args, str):
                try: args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {}
            if name:
                return ToolCall(name=name, arguments=args or {}, raw=text, format_detected="xml_tag")
        except Exception:  # noqa: BLE001, S110
            pass

    match = re.search(r'<\|tool_call>\s*call:(\w+)\s*\{(.+?)\}\s*<tool_call\|>', text, re.DOTALL)
    if match:
        name = match.group(1)
        args_str = match.group(2).strip()
        args = _extract_json_args(args_str)
        return ToolCall(name=name, arguments=args or {}, raw=text, format_detected="qwen_tool_call")

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            name = (data.get("name") or data.get("tool") or data.get("tool_name") or data.get("tool_call") or "")
            args = (data.get("arguments") or data.get("args") or data.get("parameters") or data.get("tool_call_args") or {})
            if isinstance(args, str):
                try: args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {}
            if name:
                fmt = "json_object"
                if "tool_calls" in data: fmt = "tool_calls_array"
                elif "tool_name" in data: fmt = "tool_name_key"
                elif "tool_call" in data: fmt = "legacy_tool_call"
                elif "tool" in data and "arguments" in data: fmt = "tool_key"
                return ToolCall(name=name, arguments=args or {}, raw=text, format_detected=fmt)
    except Exception:  # noqa: BLE001, S110
        pass

    match = re.search(r'"function"\s*:\s*\{[^}]*"name"\s*:\s*"([^"]+)"', text)
    if match:
        name = match.group(1)
        args_match = re.search(r'"arguments"\s*:\s*(\{[^}]+\})', text)
        args = {}
        if args_match:
            try: args = json.loads(args_match.group(1))
            except Exception:  # noqa: BLE001, S110
                pass
        return ToolCall(name=name, arguments=args, raw=text, format_detected="function_format")

    match = re.search(r'\{[^{}]*"(?:name|tool)"[^{}]*\}', text)
    if match:
        try:
            data = json.loads(match.group(0))
            name = data.get("name", data.get("tool", ""))
            args = data.get("arguments", data.get("args", {}))
            if name:
                return ToolCall(name=name, arguments=args or {}, raw=text, format_detected="regex_extract")
        except Exception:  # noqa: BLE001, S110
            pass

    return None


def parse_multiple_tool_calls(text):
    calls = []
    for match in re.finditer(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        content = match.group(1).strip()
        try:
            data = json.loads(content)
            name = data.get("name", data.get("tool", ""))
            args = data.get("arguments", data.get("args", {}))
            if isinstance(args, str):
                try: args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {}
            if name:
                calls.append(ToolCall(name=name, arguments=args or {}, raw=match.group(0), format_detected="xml_tag"))
        except Exception:  # noqa: BLE001, S110
            pass

    match = re.search(r'"tool_calls"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if match:
        try:
            items = json.loads("[" + match.group(1) + "]")
            for item in items:
                name = item.get("name", "")
                args = item.get("args", item.get("arguments", {}))
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except Exception:  # noqa: BLE001
                        args = {}
                if name:
                    calls.append(ToolCall(name=name, arguments=args or {}, raw=text, format_detected="tool_calls_array"))
        except Exception:  # noqa: BLE001, S110
            pass

    return calls
