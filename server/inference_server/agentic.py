"""Agentic loop with tool calling.

WHAT THIS FILE DOES
==================
Implements the "agent loop" — the cycle of:
  1. Send the user's request to the model
  2. If the model wants to call a tool, execute the tool
  3. Send the tool's result back to the model
  4. Repeat until the model produces a final answer

This is how LLMs can do things like "search the web, then summarize
the results" — they call a tool, get the result, then continue.

KEY CONCEPTS
============
- Tool calling: the model outputs a structured request to call a function.
- Agent loop: the iterative cycle of model → tool → result → model.
- Native vs manual parsing: some models (like Llama 3.1) have built-in
  tool support; others need manual parsing of the output text.
- Error handling: if a tool fails, what do we do? (retry, skip, fail)
"""

"""Agentic loop — handles tool calling with native llama-cpp support + fallback parsing."""
import json

from .parser import parse_tool_call


class AgenticLoop:
    """Agentic loop that handles tool calling for LLM inference.

    Supports:
    1. Native llama-cpp tool calling (create_chat_completion with tools=)
    2. Manual tool parsing from response text
    3. Custom system prompts
    4. Configurable max iterations
    """

    def __init__(
        self,
        engine,
        rag_store=None,
        embedding_model: str = "all-MiniLM-L6-v2",
        custom_system_prompt: str | None = None,
        max_iterations: int = 5,
    ):
        self.engine = engine
        self.rag_store = rag_store
        self.embedding_model = embedding_model
        self.custom_system_prompt = custom_system_prompt
        self.max_iterations = max_iterations
        self._tool_handlers = self._build_tool_handlers()

    def _build_tool_handlers(self) -> dict:
        """Build tool handler functions."""
        handlers = {}

        def rag_search(args):
            query = args.get("query", "")
            top_k = args.get("top_k", 5)
            if not self.rag_store:
                return {"error": "RAG store not available"}
            results = self.rag_store.search(
                query, top_k=top_k, embedding_model=self.embedding_model
            )
            return {
                "results": [
                    {"text": r.text[:500], "score": round(r.score, 3), "source": r.source}
                    for r in results
                ]
            }

        def calculator(args):
            expr = args.get("expression", "")
            try:
                result = eval(expr, {"__builtins__": {}}, {})
                return {"expression": expr, "result": result}
            except Exception as e:  # noqa: BLE001
                return {"expression": expr, "error": str(e)}

        def note_save(args):
            content = args.get("content", "")
            category = args.get("category", "general")
            # In a real system, save to persistent storage
            return {"saved": True, "content": content, "category": category}

        def web_search(args):
            return {
                "query": args.get("query", ""),
                "results": [],
                "note": "Web search not configured",
            }

        def rag_list(args):
            if not self.rag_store:
                return {"error": "RAG not available"}
            return {
                "documents": self.rag_store.list_documents(),
                "total_chunks": self.rag_store.count(),
            }

        handlers["rag_search"] = rag_search
        handlers["calculator"] = calculator
        handlers["note_save"] = note_save
        handlers["web_search"] = web_search
        handlers["rag_list"] = rag_list

        return handlers

    def _execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool and return the result."""
        if name in self._tool_handlers:
            try:
                result = self._tool_handlers[name](arguments)
                return json.dumps(result, ensure_ascii=False)
            except Exception as e:  # noqa: BLE001
                return json.dumps({"error": str(e)})
        return json.dumps({"error": f"Unknown tool: {name}"})

    def _build_system_prompt(self) -> str:
        """Build system prompt with tool definitions."""
        from .tools import build_tools

        tools = build_tools(None)

        prompt = self.custom_system_prompt or "You are a helpful assistant with access to tools."
        prompt += "\n\nYou have access to these tools. When you need to use one, respond with ONLY a JSON object:\n"
        prompt += '{"name": "tool_name", "arguments": {"arg": "value"}}\n\n'
        prompt += "Available tools:\n"
        for t in tools:
            fn = t["function"]
            prompt += f"- {fn['name']}: {fn['description']}\n"
        prompt += "\nIf no tool is needed, respond normally.\n"
        return prompt

    def run(self, messages: list, max_tokens: int = 512, temperature: float = 0.7) -> dict:
        """Run the agentic loop.

        Returns:
            {"response": str, "tool_calls": list, "iterations": int}
        """
        work_messages = list(messages)
        tool_log = []
        final_response = ""
        iterations = 0

        # Build system prompt with tools
        system_prompt = self._build_system_prompt()

        for iteration in range(self.max_iterations):
            iterations = iteration + 1

            # Prepare messages with system prompt
            full_messages = [{"role": "system", "content": system_prompt}] + work_messages

            # Try native tool calling first
            if self.engine.is_gguf:
                try:
                    tool_defs = []
                    for k in self._tool_handlers:
                        tool_defs.append(
                            {
                                "type": "function",
                                "function": {
                                    "name": k,
                                    "description": "Tool",
                                    "parameters": {"type": "object", "properties": {}},
                                },
                            }
                        )
                    result = self.engine.model.create_chat_completion(
                        messages=full_messages,
                        tools=tool_defs,
                        max_tokens=max_tokens,
                        temperature=max(temperature, 0.01),
                    )
                    choice = result["choices"][0]
                    msg = choice["message"]

                    if msg.get("tool_calls"):
                        tc = msg["tool_calls"][0]
                        fn = tc.get("function", {})
                        tool_name = fn.get("name", "")
                        try:
                            tool_args = json.loads(fn.get("arguments", "{}"))
                        except Exception:  # noqa: BLE001
                            tool_args = {}

                        tool_result = self._execute_tool(tool_name, tool_args)
                        tool_log.append(
                            {"tool": tool_name, "arguments": tool_args, "result": tool_result[:500]}
                        )

                        work_messages.append(
                            {"role": "assistant", "content": msg.get("content", "")}
                        )
                        work_messages.append({"role": "tool", "content": tool_result})
                        continue
                    else:
                        # No structured tool_calls — check if content has tool call text
                        content = msg.get("content", "")
                        tool_call = parse_tool_call(content)
                        if tool_call:
                            tool_result = self._execute_tool(tool_call.name, tool_call.arguments)
                            tool_log.append(
                                {
                                    "tool": tool_call.name,
                                    "arguments": tool_call.arguments,
                                    "result": tool_result[:500],
                                }
                            )
                            work_messages.append({"role": "assistant", "content": content})
                            work_messages.append({"role": "tool", "content": tool_result})
                            continue
                        final_response = content
                        break
                except Exception:  # noqa: BLE001, S110
                    pass  # Fall through to manual mode

            # Manual tool calling
            result = self.engine.generate(
                full_messages, max_tokens=max_tokens, temperature=temperature
            )
            content = result if isinstance(result, str) else result.get("response", str(result))

            tool_call = parse_tool_call(content)
            if tool_call:
                tool_result = self._execute_tool(tool_call.name, tool_call.arguments)
                tool_log.append(
                    {
                        "tool": tool_call.name,
                        "arguments": tool_call.arguments,
                        "result": tool_result[:500],
                    }
                )

                work_messages.append({"role": "assistant", "content": content})
                work_messages.append({"role": "tool", "content": tool_result})
                continue
            else:
                final_response = content
                break

        return {
            "response": final_response,
            "tool_calls": tool_log,
            "iterations": iterations,
        }
