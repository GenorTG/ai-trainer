"""inference_server — lightweight inference server with RAG, tool calling, MCP.

This package is the CANONICAL SOURCE for:
- Jinja2 template rendering (templates/renderer.py)
- Tool call parsing (parser.py)
- Built-in tools (tools.py)
- RAG store and ingestor (rag.py)
- Document parsers for 33 formats (parsers.py)
- FastAPI server (server.py, server_v2.py)
- Agentic loop with tool calling (agentic.py)

finetune-studio DEPENDS on this package for template rendering and tool parsing.
"""

"""Portable Inference Server with RAG — standalone deployment package."""
