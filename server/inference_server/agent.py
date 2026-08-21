"""High-level agent abstraction.

WHAT THIS FILE DOES
==================
Wraps the inference engine with conversation state management. The
agent keeps track of the conversation history and provides a simple
.send() method to continue the dialogue.

KEY CONCEPTS
============
- Conversation state: the agent remembers previous messages.
- Stateless vs stateful: a stateless API treats each request independently;
  a stateful agent maintains context across calls.
- Token budgets: limiting how much conversation history to include in
  each request (older messages get dropped).
"""

"""Agent loop — handles tool calling for models that support it."""
import json

from .tools import build_tools, execute_tool


class ToolCallingAgent:
    """Wraps an inference engine with tool-calling capability.

    Supports two modes:
    1. Native tool calling (llama.cpp / GGUF models with chat templates that support tools)
    2. Manual tool parsing (any model — we parse the response for tool-call JSON)
    """

    def __init__(
        self,
        engine,
        rag_store,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_iterations: int = 5,
        native_tools: bool = True,
    ):
        self.engine = engine
        self.rag_store = rag_store
        self.embedding_model = embedding_model
        self.max_iterations = max_iterations
        self.native_tools = native_tools and engine.is_gguf
        self.tools = build_tools(rag_store, embedding_model)

    def _format_tool_prompt(self) -> str:
        """Build a system prompt describing available tools (for manual mode).

        Uses Qwen-style <tool_call> format since small Qwen models are trained on it.
        """
        desc = []
        for t in self.tools:
            fn = t["function"]
            desc.append(f"- {fn['name']}: {fn['description']}")
        return (
            "You have access to the following tools. When you need to use one, "
            "respond with EXACTLY this format (no other text):\n"
            '<tool_call>{"name": "tool_name", "arguments": {"arg": "value"}}</tool_call>\n\n'
            "Available tools:\n" + "\n".join(desc) + "\n\n"
            "After the tool result is provided, answer the user normally."
            "If no tool is needed, respond normally."
        )

    def _parse_manual_tool_call(self, text: str) -> dict | None:
        """Try to extract a tool call from model output (manual mode).

        Handles formats:
        - {"tool": "name", "arguments": {...}} (agent format)
        - {"name": "name", "arguments": {...}} (OpenAI-style)
        - <tool_call>{"name": ..., "arguments": ...}</tool_call> (Qwen/chatml)
        - <tool_call>{{...}}</tool_call> (Qwen with Jinja double-brace wrapper)
        - ```json ... ``` blocks
        """
        text = text.strip()

        # Strip tool_call tags
        text = text.replace("<tool_call>", "").replace("</tool_call>", "")

        # Find outermost JSON via brace matching
        candidates = self._extract_json_objects(text)

        for candidate in candidates:
            # Try as-is
            parsed = self._try_parse(candidate)
            if parsed:
                return parsed
            # Try unwrapping one layer of double braces: {{...}} -> {...}
            if len(candidate) >= 2 and candidate.startswith("{") and candidate.endswith("}"):
                unwrapped = candidate[1:-1]
                parsed = self._try_parse(unwrapped)
                if parsed:
                    return parsed
        return None

    def _extract_json_objects(self, text: str) -> list[str]:
        """Extract all balanced JSON objects from text."""
        candidates = []
        for i, ch in enumerate(text):
            if ch != "{":
                continue
            depth = 0
            in_string = False
            escape = False
            for j in range(i, len(text)):
                c = text[j]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[i : j + 1])
                        break
        return candidates

    def _try_parse(self, text: str) -> dict | None:
        """Try to parse text as a tool-call dict."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        if "tool" in data or "name" in data:
            return self._normalize_tool_call(data)
        return None

    def _normalize_tool_call(self, data: dict) -> dict:
        """Normalize different tool-call formats to {"tool": ..., "arguments": {...}}."""
        if "tool" in data:
            args = data.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return {"tool": data["tool"], "arguments": args}
        if "name" in data:
            args = data.get("arguments", data.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return {"tool": data["name"], "arguments": args}
        return data

    def run(
        self,
        messages: list,
        max_tokens: int = 512,
        temperature: float = 0.7,
        max_iterations: int | None = None,
    ) -> dict:
        """Run the agent loop. Returns final response + tool call log."""
        iterations = max_iterations or self.max_iterations
        work_messages = [dict(m) for m in messages]
        tool_log = []
        final_response = ""
        tool_calls_made = 0

        for _ in range(iterations):
            # Generate
            if self.native_tools:
                result = self._generate_native(work_messages, max_tokens, temperature)
            else:
                result = self._generate_manual(work_messages, max_tokens, temperature)

            content = result["response"]

            # Check if this is a tool call
            tool_call = None
            if self.native_tools:
                tool_call = result.get("tool_call")
                # llama.cpp may return the tool call as raw text content
                # instead of structured tool_calls — parse it manually
                if tool_call is None and content:
                    tool_call = self._parse_manual_tool_call(content)
            else:
                tool_call = self._parse_manual_tool_call(content)

            if tool_call is None:
                # Normal response
                final_response = content
                break

            # Execute tool
            tool_name = tool_call.get("tool", tool_call.get("name", ""))
            arguments = tool_call.get("arguments", tool_call.get("parameters", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            tool_result = execute_tool(tool_name, arguments, self.rag_store, self.embedding_model)
            tool_log.append(
                {
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": tool_result[:2000],
                    "iteration": tool_calls_made + 1,
                }
            )
            tool_calls_made += 1

            # Add tool result to messages
            if self.native_tools:
                work_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.get("tool_call_id", f"call_{tool_calls_made}"),
                        "content": tool_result,
                    }
                )
            else:
                work_messages.append(
                    {
                        "role": "user",
                        "content": f"[Tool {tool_name} result]: {tool_result}\n\nNow continue with your answer, or call another tool if needed.",
                    }
                )

        return {
            "response": final_response or content,
            "tool_calls": tool_log,
            "iterations": tool_calls_made + 1,
        }

    def _generate_native(self, messages, max_tokens, temperature):
        """Generate using llama.cpp native tool calling."""
        try:
            output = self.engine.model.create_chat_completion(
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
            )
            choice = output["choices"][0]
            msg = choice["message"]

            if msg.get("tool_calls"):
                tc = msg["tool_calls"][0]
                fn = tc.get("function", {})
                return {
                    "response": msg.get("content") or "",
                    "tool_call": {
                        "tool": fn.get("name", ""),
                        "arguments": json.loads(fn.get("arguments", "{}") or "{}"),
                    },
                    "tool_call_id": tc.get("id", "call_1"),
                }
            return {"response": msg.get("content", "")}

        except Exception:  # noqa: BLE001
            # Fall back to manual mode
            self.native_tools = False
            return self._generate_manual(messages, max_tokens, temperature)

    def _generate_manual(self, messages, max_tokens, temperature):
        """Generate with tool instructions in system prompt."""
        work = [dict(m) for m in messages]
        # Inject tool instructions into system prompt
        tool_prompt = self._format_tool_prompt()
        if work and work[0]["role"] == "system":
            work[0]["content"] = work[0]["content"] + "\n\n" + tool_prompt
        else:
            work.insert(0, {"role": "system", "content": tool_prompt})

        response = self.engine.generate(work, max_tokens=max_tokens, temperature=temperature)
        return {"response": response}
