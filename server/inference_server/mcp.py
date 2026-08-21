"""Model Context Protocol (MCP) server.

WHAT THIS FILE DOES
==================
Implements the Model Context Protocol — a standard way for LLMs to
discover and call tools. With MCP, external clients (like Claude
Desktop or other MCP-compatible tools) can connect to our server
and use the tools we expose.

KEY CONCEPTS
============
- MCP: a protocol (like HTTP) for LLM-tool communication. Developed
  by Anthropic, now an open standard.
- Server vs client: we are the server; MCP clients (other apps) connect to us.
- Tool discovery: clients can ask "what tools do you have?" and we
  respond with the list.
- Tool invocation: clients send a tool call, we execute it and return the result.
- JSON-RPC: the underlying protocol MCP uses for communication.
"""

"""MCP (Model Context Protocol) support for inference server."""
from collections.abc import Callable
from dataclasses import dataclass, field
import json


@dataclass
class MCPTool:
    """MCP tool definition."""

    name: str
    description: str
    input_schema: dict = field(default_factory=dict)
    handler: Callable = None

    def to_openai_format(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass
class MCPServer:
    """MCP server — manages tools and handles requests."""

    name: str = "default"
    tools: dict = field(default_factory=dict)
    description: str = ""

    def register_tool(self, tool: MCPTool):
        """Register a tool."""
        self.tools[tool.name] = tool

    def list_tools(self) -> list:
        """List all registered tools."""
        return [t.to_openai_format() for t in self.tools.values()]

    def execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool and return the result."""
        if name not in self.tools:
            return json.dumps({"error": f"Unknown tool: {name}"})

        tool = self.tools[name]
        if tool.handler:
            try:
                result = tool.handler(arguments)
                return json.dumps(result) if not isinstance(result, str) else result
            except Exception as e:  # noqa: BLE001
                return json.dumps({"error": str(e)})
        return json.dumps({"error": f"Tool {name} has no handler"})

    def to_dict(self) -> dict:
        """Serialize server info."""
        return {
            "name": self.name,
            "description": self.description,
            "tools": [{"name": t.name, "description": t.description} for t in self.tools.values()],
        }


class RAGMCPServer(MCPServer):
    """MCP server with RAG tools built-in."""

    def __init__(self, rag_store=None, embedding_model: str = "all-MiniLM-L6-v2"):
        super().__init__(name="rag_server", description="RAG-enabled inference server")
        self.rag_store = rag_store
        self.embedding_model = embedding_model
        self._register_default_tools()

    def _register_default_tools(self):
        self.register_tool(
            MCPTool(
                name="rag_search",
                description="Search the knowledge base for relevant information",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                    "required": ["query"],
                },
                handler=self._rag_search,
            )
        )
        self.register_tool(
            MCPTool(
                name="rag_ingest",
                description="Add a document to the knowledge base",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path"],
                },
                handler=self._rag_ingest,
            )
        )
        self.register_tool(
            MCPTool(
                name="rag_list",
                description="List all documents in the knowledge base",
                input_schema={"type": "object", "properties": {}},
                handler=self._rag_list,
            )
        )
        self.register_tool(
            MCPTool(
                name="web_search",
                description="Search the web for information",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=self._web_search,
            )
        )
        self.register_tool(
            MCPTool(
                name="calculator",
                description="Calculate a math expression",
                input_schema={
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
                handler=self._calculator,
            )
        )

    def _rag_search(self, args: dict) -> dict:
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        if not self.rag_store:
            return {"error": "RAG store not initialized"}
        results = self.rag_store.search(query, top_k=top_k, embedding_model=self.embedding_model)
        return {
            "results": [
                {"text": r.text[:500], "score": round(r.score, 3), "source": r.source}
                for r in results
            ]
        }

    def _rag_ingest(self, args: dict) -> dict:
        path = args.get("path", "")
        content = args.get("content", "")
        if not path and not content:
            return {"error": "No path or content provided"}
        if not self.rag_store:
            return {"error": "RAG store not initialized"}

        from .ingest import ingest_bytes, ingest_file

        if content:
            result = ingest_bytes(path or "api_upload", content.encode(), self.rag_store)
        else:
            result = ingest_file(path, store=self.rag_store)
        return result

    def _rag_list(self, args: dict) -> dict:
        if not self.rag_store:
            return {"error": "RAG store not initialized"}
        return {
            "documents": self.rag_store.list_documents(),
            "total_chunks": self.rag_store.count(),
        }

    def _web_search(self, args: dict) -> dict:
        query = args.get("query", "")
        return {
            "query": query,
            "results": [],
            "note": "Web search not configured — add API key in config",
        }

    def _calculator(self, args: dict) -> dict:
        expr = args.get("expression", "")
        try:
            result = eval(expr, {"__builtins__": {}}, {})
            return {"expression": expr, "result": result}
        except Exception as e:  # noqa: BLE001
            return {"expression": expr, "error": str(e)}
