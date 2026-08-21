"""Built-in tools for the agentic loop.

WHAT THIS FILE DOES
==================
Provides the tools that the model can call during conversations:
  - rag_search: search the RAG store for relevant context
  - rag_ingest: add a document to the RAG store
  - rag_list: list all documents in the store
  - calculator: evaluate a math expression
  - web_search: search the web (optional, requires API key)

KEY CONCEPTS
============
- Tool interface: each tool is a function that takes a dict of arguments
  and returns a string (the result sent back to the model).
- Tool registration: tools are listed in a registry so the model can
  be told what tools are available.
- Argument parsing: tool arguments come from the model as JSON. We
  parse them and pass to the tool function.
- Error handling: if a tool fails, we return an error message to the
  model so it can try a different approach.
"""

"""Agentic RAG tools — tools the model can call during conversation."""
import json

from .parsers import parse_document
from .rag import RAGStore


def build_tools(rag_store: RAGStore, embedding_model: str = "all-MiniLM-L6-v2") -> list[dict]:
    """Build tool definitions for the model."""
    return [
        {
            "type": "function",
            "function": {
                "name": "rag_search",
                "description": "Search indexed documents for relevant information. Use when the user asks about company data, projects, documents, or anything that might be in the knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results (default 5)",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rag_list_documents",
                "description": "List all documents currently in the knowledge base.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rag_ingest_document",
                "description": "Add a document to the knowledge base so it can be queried later. Use when the user provides or references a file/document.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the document file"}
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rag_remove_document",
                "description": "Remove a document from the knowledge base.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string", "description": "Document ID to remove"}
                    },
                    "required": ["document_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rag_stats",
                "description": "Get statistics about the knowledge base (document count, chunk count).",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "parse_document",
                "description": "Parse a document file (PDF, DOCX, XLSX, PPTX, etc.) and return its text content. Use when the user asks about the contents of a specific file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the document"}
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "save_conversation_knowledge",
                "description": "Save an important fact from the conversation to the knowledge base, so it's remembered in future sessions. Use when the user shares important information about themselves, their company, or a project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The fact/information to remember",
                        }
                    },
                    "required": ["content"],
                },
            },
        },
    ]


def execute_tool(
    tool_name: str, arguments: dict, rag_store: RAGStore, embedding_model: str = "all-MiniLM-L6-v2"
) -> str:
    """Execute a tool call and return the result."""
    try:
        if tool_name == "rag_search":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", 5)
            results = rag_store.search(query, top_k=top_k, embedding_model=embedding_model)
            if not results:
                return json.dumps(
                    {"results": [], "message": "No relevant documents found"}, ensure_ascii=False
                )
            return json.dumps(
                {
                    "results": [
                        {
                            "text": r.text,
                            "score": round(r.score, 3),
                            "source": r.source,
                            "document_id": r.document_id,
                        }
                        for r in results
                    ]
                },
                ensure_ascii=False,
            )

        elif tool_name == "rag_list_documents":
            docs = rag_store.list_documents()
            return json.dumps(
                {"documents": docs, "total_chunks": rag_store.count()}, ensure_ascii=False
            )

        elif tool_name == "rag_ingest_document":
            path = arguments.get("path", "")
            from .rag import DocumentIngestor

            ingestor = DocumentIngestor(rag_store)
            result = ingestor.ingest_file(path, embedding_model=embedding_model)
            return json.dumps(result, ensure_ascii=False)

        elif tool_name == "rag_remove_document":
            doc_id = arguments.get("document_id", "")
            removed = rag_store.remove_document(doc_id)
            return json.dumps(
                {"document_id": doc_id, "chunks_removed": removed}, ensure_ascii=False
            )

        elif tool_name == "rag_stats":
            return json.dumps(
                {
                    "total_chunks": rag_store.count(),
                    "total_documents": len(rag_store.list_documents()),
                    "documents": rag_store.list_documents(),
                },
                ensure_ascii=False,
            )

        elif tool_name == "parse_document":
            path = arguments.get("path", "")
            text = parse_document(path)
            return text[:8000] if len(text) > 8000 else text

        elif tool_name == "save_conversation_knowledge":
            content = arguments.get("content", "")
            import hashlib

            doc_id = hashlib.md5(f"memory:{content[:50]}".encode()).hexdigest()[:12]
            words = content.split()
            chunks = []
            start = 0
            idx = 0
            while start < len(words):
                end = min(start + 200, len(words))
                chunks.append(
                    {
                        "id": f"{doc_id}_{idx}",
                        "text": " ".join(words[start:end]),
                        "source": "conversation_memory",
                    }
                )
                idx += 1
                start += 150
            added = rag_store.add_document(doc_id, chunks, embedding_model)
            return json.dumps(
                {
                    "saved": True,
                    "document_id": doc_id,
                    "chunks": added,
                    "message": "Knowledge saved. It will be available in future queries.",
                },
                ensure_ascii=False,
            )

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)

    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": str(e)}, ensure_ascii=False)
