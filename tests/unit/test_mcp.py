"""Tests for inference_server/mcp.py."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestMCPTool:
    def test_init(self):
        from inference_server.mcp import MCPTool
        tool = MCPTool(name="search", description="Search the web")
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert tool.input_schema == {}
        assert tool.handler is None

    def test_to_openai_format(self):
        from inference_server.mcp import MCPTool
        tool = MCPTool(
            name="calculator",
            description="Calculate",
            input_schema={"type": "object", "properties": {"expr": {"type": "string"}}},
        )
        fmt = tool.to_openai_format()
        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "calculator"
        assert fmt["function"]["description"] == "Calculate"
        assert "expr" in fmt["function"]["parameters"]["properties"]

    def test_to_openai_format_empty_schema(self):
        from inference_server.mcp import MCPTool
        tool = MCPTool(name="noop", description="No op")
        fmt = tool.to_openai_format()
        assert fmt["function"]["parameters"] == {}


@pytest.mark.unit
class TestMCPServer:
    def test_init(self):
        from inference_server.mcp import MCPServer
        srv = MCPServer()
        assert srv.name == "default"
        assert srv.tools == {}

    def test_init_custom(self):
        from inference_server.mcp import MCPServer
        srv = MCPServer(name="my_server", description="Custom")
        assert srv.name == "my_server"
        assert srv.description == "Custom"

    def test_register_tool(self):
        from inference_server.mcp import MCPServer, MCPTool
        srv = MCPServer()
        tool = MCPTool(name="a", description="Tool A")
        srv.register_tool(tool)
        assert "a" in srv.tools
        assert srv.tools["a"] is tool

    def test_list_tools(self):
        from inference_server.mcp import MCPServer, MCPTool
        srv = MCPServer()
        srv.register_tool(MCPTool(name="x", description="X"))
        srv.register_tool(MCPTool(name="y", description="Y"))
        tools = srv.list_tools()
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"x", "y"}

    def test_execute_tool_success(self):
        from inference_server.mcp import MCPServer, MCPTool
        srv = MCPServer()
        srv.register_tool(MCPTool(
            name="echo",
            description="Echo",
            handler=lambda args: {"echoed": args.get("msg", "")},
        ))
        result = srv.execute_tool("echo", {"msg": "hello"})
        assert "hello" in result

    def test_execute_tool_no_handler(self):
        from inference_server.mcp import MCPServer, MCPTool
        srv = MCPServer()
        srv.register_tool(MCPTool(name="broken", description="No handler"))
        result = srv.execute_tool("broken", {})
        assert "error" in result
        assert "no handler" in result.lower()

    def test_execute_unknown_tool(self):
        from inference_server.mcp import MCPServer
        srv = MCPServer()
        result = srv.execute_tool("nonexistent", {})
        assert "error" in result
        assert "Unknown tool" in result

    def test_execute_tool_handler_error(self):
        from inference_server.mcp import MCPServer, MCPTool
        srv = MCPServer()

        def bad_handler(args):
            raise ValueError("boom")

        srv.register_tool(MCPTool(name="fail", description="Fails", handler=bad_handler))
        result = srv.execute_tool("fail", {})
        assert "error" in result
        assert "boom" in result

    def test_execute_tool_returns_string(self):
        from inference_server.mcp import MCPServer, MCPTool
        srv = MCPServer()
        srv.register_tool(MCPTool(
            name="str_tool",
            description="Returns string",
            handler=lambda args: "raw string result",
        ))
        result = srv.execute_tool("str_tool", {})
        assert result == "raw string result"

    def test_to_dict(self):
        from inference_server.mcp import MCPServer, MCPTool
        srv = MCPServer(name="test", description="Test server")
        srv.register_tool(MCPTool(name="a", description="Tool A"))
        d = srv.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "Test server"
        assert len(d["tools"]) == 1
        assert d["tools"][0]["name"] == "a"


@pytest.mark.unit
class TestRAGMCPServer:
    def test_init_registers_default_tools(self, mocker):
        from inference_server.mcp import RAGMCPServer
        srv = RAGMCPServer(rag_store=mocker.MagicMock())
        tool_names = set(srv.tools.keys())
        assert "rag_search" in tool_names
        assert "rag_ingest" in tool_names
        assert "rag_list" in tool_names
        assert "web_search" in tool_names
        assert "calculator" in tool_names

    def test_rag_search(self, mocker):
        from inference_server.mcp import RAGMCPServer
        mock_store = mocker.MagicMock()
        mock_result = mocker.MagicMock(text="chunk", score=0.9, source="doc.txt")
        mock_store.search.return_value = [mock_result]
        srv = RAGMCPServer(rag_store=mock_store)
        result = srv._rag_search({"query": "test", "top_k": 1})
        assert "results" in result
        assert len(result["results"]) == 1

    def test_rag_search_no_store(self):
        from inference_server.mcp import RAGMCPServer
        srv = RAGMCPServer(rag_store=None)
        result = srv._rag_search({"query": "test"})
        assert "error" in result

    def test_rag_ingest_no_store(self):
        from inference_server.mcp import RAGMCPServer
        srv = RAGMCPServer(rag_store=None)
        result = srv._rag_ingest({"path": "test.txt"})
        assert "error" in result

    def test_rag_ingest_no_content(self, mocker):
        from inference_server.mcp import RAGMCPServer
        srv = RAGMCPServer(rag_store=mocker.MagicMock())
        result = srv._rag_ingest({})
        assert "error" in result

    def test_rag_list_no_store(self):
        from inference_server.mcp import RAGMCPServer
        srv = RAGMCPServer(rag_store=None)
        result = srv._rag_list({})
        assert "error" in result

    def test_rag_list_with_store(self, mocker):
        from inference_server.mcp import RAGMCPServer
        mock_store = mocker.MagicMock()
        mock_store.list_documents.return_value = [{"document_id": "d1", "chunk_count": 3}]
        mock_store.count.return_value = 3
        srv = RAGMCPServer(rag_store=mock_store)
        result = srv._rag_list({})
        assert "documents" in result
        assert result["total_chunks"] == 3

    def test_web_search(self):
        from inference_server.mcp import RAGMCPServer
        srv = RAGMCPServer(rag_store=None)
        result = srv._web_search({"query": "test"})
        assert result["query"] == "test"
        assert "note" in result

    def test_calculator_valid(self):
        from inference_server.mcp import RAGMCPServer
        srv = RAGMCPServer(rag_store=None)
        result = srv._calculator({"expression": "2 + 3"})
        assert result["result"] == 5

    def test_calculator_invalid(self):
        from inference_server.mcp import RAGMCPServer
        srv = RAGMCPServer(rag_store=None)
        result = srv._calculator({"expression": "import os"})
        assert "error" in result

    def test_calculator_empty(self):
        from inference_server.mcp import RAGMCPServer
        srv = RAGMCPServer(rag_store=None)
        result = srv._calculator({})
        assert "error" in result
