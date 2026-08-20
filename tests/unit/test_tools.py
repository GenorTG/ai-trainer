"""Tests for inference_server.tools — tool registry + execution.

Covers build_tools() and execute_tool() with full mock-based coverage.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from inference_server.tools import build_tools, execute_tool


# =============================================================================
# build_tools tests
# =============================================================================


class TestBuildTools:
    """Tests for build_tools() — tool schema generation."""

    def test_returns_list_of_tool_definitions(self):
        """build_tools returns a list of tool definitions."""
        rag_store = MagicMock()
        tools = build_tools(rag_store)

        assert isinstance(tools, list)
        assert len(tools) >= 5  # We define 7 tools

    def test_all_tools_have_function_type(self):
        """Each tool definition has type='function'."""
        rag_store = MagicMock()
        tools = build_tools(rag_store)

        for tool in tools:
            assert tool["type"] == "function"

    def test_all_tools_have_name_description_parameters(self):
        """Each tool has name, description, parameters."""
        rag_store = MagicMock()
        tools = build_tools(rag_store)

        for tool in tools:
            fn = tool["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_required_params_have_required_field(self):
        """Tools that need args have 'required' field listing them."""
        rag_store = MagicMock()
        tools = build_tools(rag_store)

        # rag_search requires query
        rag_search = next(t for t in tools if t["function"]["name"] == "rag_search")
        assert "query" in rag_search["function"]["parameters"]["required"]

    def test_optional_params_have_descriptions(self):
        """All parameter properties have descriptions."""
        rag_store = MagicMock()
        tools = build_tools(rag_store)

        for tool in tools:
            params = tool["function"]["parameters"]["properties"]
            for param_name, param_schema in params.items():
                assert "description" in param_schema, f"Missing description for {param_name}"

    def test_embedding_model_is_optional(self):
        """embedding_model parameter has default value."""
        rag_store = MagicMock()
        tools = build_tools(rag_store)
        # Should not raise
        assert tools is not None

    def test_custom_embedding_model_accepted(self):
        """Custom embedding_model name works."""
        rag_store = MagicMock()
        tools = build_tools(rag_store, embedding_model="custom-model-v2")
        assert tools is not None

    def test_tool_names_are_unique(self):
        """No duplicate tool names."""
        rag_store = MagicMock()
        tools = build_tools(rag_store)
        names = [t["function"]["name"] for t in tools]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"


# =============================================================================
# execute_tool tests — rag_search
# =============================================================================


class TestExecuteToolRagSearch:
    """Tests for execute_tool(rag_search)."""

    def test_rag_search_with_results(self):
        """rag_search returns JSON-formatted results."""
        rag_store = MagicMock()
        rag_store.search.return_value = [
            MagicMock(text="Fact 1", score=0.95, source="doc1.md", document_id="d1"),
            MagicMock(text="Fact 2", score=0.85, source="doc2.md", document_id="d2"),
        ]
        result = execute_tool("rag_search", {"query": "test"}, rag_store)

        parsed = json.loads(result)
        assert "results" in parsed
        assert len(parsed["results"]) == 2
        assert parsed["results"][0]["text"] == "Fact 1"
        assert parsed["results"][0]["score"] == 0.95

    def test_rag_search_no_results_returns_empty(self):
        """rag_search with no results returns empty list + message."""
        rag_store = MagicMock()
        rag_store.search.return_value = []
        result = execute_tool("rag_search", {"query": "nothing"}, rag_store)

        parsed = json.loads(result)
        assert parsed["results"] == []
        assert "message" in parsed

    def test_rag_search_uses_top_k_default(self):
        """rag_search defaults top_k to 5."""
        rag_store = MagicMock()
        rag_store.search.return_value = []
        execute_tool("rag_search", {"query": "test"}, rag_store)

        rag_store.search.assert_called_once()
        call_kwargs = rag_store.search.call_args.kwargs
        assert call_kwargs["top_k"] == 5

    def test_rag_search_respects_top_k(self):
        """rag_search uses provided top_k."""
        rag_store = MagicMock()
        rag_store.search.return_value = []
        execute_tool("rag_search", {"query": "test", "top_k": 10}, rag_store)

        call_kwargs = rag_store.search.call_args.kwargs
        assert call_kwargs["top_k"] == 10

    def test_rag_search_uses_provided_embedding_model(self):
        """rag_search passes embedding_model through."""
        rag_store = MagicMock()
        rag_store.search.return_value = []
        execute_tool("rag_search", {"query": "test"}, rag_store, embedding_model="custom-model")

        call_kwargs = rag_store.search.call_args.kwargs
        assert call_kwargs["embedding_model"] == "custom-model"

    def test_rag_search_handles_exception(self):
        """rag_search catches store exceptions."""
        rag_store = MagicMock()
        rag_store.search.side_effect = RuntimeError("DB error")
        result = execute_tool("rag_search", {"query": "test"}, rag_store)

        parsed = json.loads(result)
        assert "error" in parsed
        assert "DB error" in parsed["error"]


# =============================================================================
# execute_tool tests — rag_list_documents
# =============================================================================


class TestExecuteToolRagListDocuments:
    """Tests for execute_tool(rag_list_documents)."""

    def test_rag_list_documents_returns_list(self):
        """Returns documents list + total_chunks."""
        rag_store = MagicMock()
        rag_store.list_documents.return_value = ["doc1", "doc2"]
        rag_store.count.return_value = 42

        result = execute_tool("rag_list_documents", {}, rag_store)
        parsed = json.loads(result)

        assert parsed["documents"] == ["doc1", "doc2"]
        assert parsed["total_chunks"] == 42

    def test_rag_list_documents_empty(self):
        """Returns empty list when store is empty."""
        rag_store = MagicMock()
        rag_store.list_documents.return_value = []
        rag_store.count.return_value = 0

        result = execute_tool("rag_list_documents", {}, rag_store)
        parsed = json.loads(result)

        assert parsed["documents"] == []
        assert parsed["total_chunks"] == 0


# =============================================================================
# execute_tool tests — rag_ingest_document
# =============================================================================


class TestExecuteToolRagIngest:
    """Tests for execute_tool(rag_ingest_document)."""

    def test_rag_ingest_calls_ingestor(self, tmp_path):
        """rag_ingest_document calls DocumentIngestor.ingest_file."""
        rag_store = MagicMock()
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test\nContent here")

        with patch("inference_server.tools.DocumentIngestor") as MockIngestor:
            mock_ingestor_instance = MagicMock()
            mock_ingestor_instance.ingest_file.return_value = {
                "document_id": "abc123",
                "chunks": 5,
            }
            MockIngestor.return_value = mock_ingestor_instance

            result = execute_tool(
                "rag_ingest_document",
                {"path": str(test_file)},
                rag_store,
            )

        parsed = json.loads(result)
        assert parsed["document_id"] == "abc123"
        assert parsed["chunks"] == 5

    def test_rag_ingest_passes_embedding_model(self, tmp_path):
        """rag_ingest_document passes embedding_model to ingestor."""
        rag_store = MagicMock()
        test_file = tmp_path / "test.md"
        test_file.write_text("Content")

        with patch("inference_server.tools.DocumentIngestor") as MockIngestor:
            mock_ingestor_instance = MagicMock()
            mock_ingestor_instance.ingest_file.return_value = {"document_id": "x"}
            MockIngestor.return_value = mock_ingestor_instance

            execute_tool(
                "rag_ingest_document",
                {"path": str(test_file)},
                rag_store,
                embedding_model="custom-emb",
            )

        mock_ingestor_instance.ingest_file.assert_called_once()
        call_kwargs = mock_ingestor_instance.ingest_file.call_args.kwargs
        assert call_kwargs["embedding_model"] == "custom-emb"


# =============================================================================
# execute_tool tests — rag_remove_document
# =============================================================================


class TestExecuteToolRagRemove:
    """Tests for execute_tool(rag_remove_document)."""

    def test_rag_remove_removes_chunks(self):
        """rag_remove_document calls remove_document."""
        rag_store = MagicMock()
        rag_store.remove_document.return_value = 5

        result = execute_tool("rag_remove_document", {"document_id": "doc1"}, rag_store)
        parsed = json.loads(result)

        assert parsed["document_id"] == "doc1"
        assert parsed["chunks_removed"] == 5

    def test_rag_remove_zero_chunks(self):
        """Returns chunks_removed=0 when doc not found."""
        rag_store = MagicMock()
        rag_store.remove_document.return_value = 0

        result = execute_tool("rag_remove_document", {"document_id": "missing"}, rag_store)
        parsed = json.loads(result)

        assert parsed["chunks_removed"] == 0


# =============================================================================
# execute_tool tests — rag_stats
# =============================================================================


class TestExecuteToolRagStats:
    """Tests for execute_tool(rag_stats)."""

    def test_rag_stats_returns_counts(self):
        """rag_stats returns total_chunks + total_documents + list."""
        rag_store = MagicMock()
        rag_store.count.return_value = 100
        rag_store.list_documents.return_value = ["doc1", "doc2", "doc3"]

        result = execute_tool("rag_stats", {}, rag_store)
        parsed = json.loads(result)

        assert parsed["total_chunks"] == 100
        assert parsed["total_documents"] == 3
        assert parsed["documents"] == ["doc1", "doc2", "doc3"]


# =============================================================================
# execute_tool tests — parse_document
# =============================================================================


class TestExecuteToolParseDocument:
    """Tests for execute_tool(parse_document)."""

    def test_parse_document_returns_text(self, tmp_path):
        """parse_document tool returns parsed text content."""
        from inference_server.parsers import parse_document as real_parse

        test_file = tmp_path / "test.md"
        test_file.write_text("# Title\n\nBody text here")

        rag_store = MagicMock()
        result = execute_tool("parse_document", {"path": str(test_file)}, rag_store)

        # Returns actual parsed text (function delegates to parsers.parse_document)
        assert "Title" in result or "Body" in result

    def test_parse_document_truncates_long_text(self, tmp_path):
        """parse_document truncates to 8000 chars."""
        test_file = tmp_path / "long.md"
        # Write 10k chars
        test_file.write_text("x" * 10000)

        rag_store = MagicMock()
        result = execute_tool("parse_document", {"path": str(test_file)}, rag_store)

        assert len(result) <= 8000


# =============================================================================
# execute_tool tests — save_conversation_knowledge
# =============================================================================


class TestExecuteToolSaveConversationKnowledge:
    """Tests for execute_tool(save_conversation_knowledge)."""

    def test_save_knowledge_creates_document_id(self):
        """save_conversation_knowledge creates MD5-based doc_id."""
        rag_store = MagicMock()
        rag_store.add_document.return_value = 3

        result = execute_tool(
            "save_conversation_knowledge",
            {"content": "Important fact"},
            rag_store,
        )
        parsed = json.loads(result)

        assert parsed["saved"] is True
        assert "document_id" in parsed
        assert len(parsed["document_id"]) == 12  # MD5 truncated

    def test_save_knowledge_chunks_text(self):
        """Saves content with overlap chunks of 200 words, 50 step."""
        rag_store = MagicMock()
        rag_store.add_document.return_value = 1

        # 500 words = 3 chunks at 200 size, 150 overlap
        long_content = " ".join([f"word{i}" for i in range(500)])
        execute_tool(
            "save_conversation_knowledge",
            {"content": long_content},
            rag_store,
        )

        rag_store.add_document.assert_called_once()
        call_args = rag_store.add_document.call_args
        chunks = call_args.args[1]  # second positional arg is chunks
        assert len(chunks) >= 2  # 500 words / 150 step ≈ 3.3 chunks

    def test_save_knowledge_uses_conversation_memory_source(self):
        """All chunks have source='conversation_memory'."""
        rag_store = MagicMock()
        rag_store.add_document.return_value = 1

        execute_tool(
            "save_conversation_knowledge",
            {"content": "Fact content"},
            rag_store,
        )

        call_args = rag_store.add_document.call_args
        chunks = call_args.args[1]
        for chunk in chunks:
            assert chunk["source"] == "conversation_memory"

    def test_save_knowledge_uses_embedding_model(self):
        """save_conversation_knowledge passes embedding_model to add_document."""
        rag_store = MagicMock()
        rag_store.add_document.return_value = 1

        execute_tool(
            "save_conversation_knowledge",
            {"content": "Test"},
            rag_store,
            embedding_model="custom-emb",
        )

        call_kwargs = rag_store.add_document.call_args.kwargs
        assert call_kwargs["embedding_model"] == "custom-emb"

    def test_save_knowledge_short_content_one_chunk(self):
        """Content < 200 words = 1 chunk."""
        rag_store = MagicMock()
        rag_store.add_document.return_value = 1

        execute_tool(
            "save_conversation_knowledge",
            {"content": "short content"},
            rag_store,
        )

        call_args = rag_store.add_document.call_args
        chunks = call_args.args[1]
        assert len(chunks) == 1


# =============================================================================
# execute_tool tests — unknown tool
# =============================================================================


class TestExecuteToolUnknown:
    """Tests for execute_tool on unknown tool name."""

    def test_unknown_tool_returns_error(self):
        """Returns error JSON for unknown tool."""
        rag_store = MagicMock()
        result = execute_tool("not_a_real_tool", {}, rag_store)

        parsed = json.loads(result)
        assert "error" in parsed
        assert "Unknown tool" in parsed["error"]

    def test_tool_exception_caught(self):
        """Exceptions are caught and returned as JSON error."""
        rag_store = MagicMock()
        rag_store.search.side_effect = RuntimeError("DB down")

        result = execute_tool("rag_search", {"query": "x"}, rag_store)

        parsed = json.loads(result)
        assert "error" in parsed
        assert "DB down" in parsed["error"]


# =============================================================================
# JSON output format
# =============================================================================


class TestToolOutputFormat:
    """Tests that tool outputs are valid JSON."""

    @pytest.mark.parametrize("tool_name,args", [
        ("rag_search", {"query": "test"}),
        ("rag_list_documents", {}),
        ("rag_stats", {}),
        ("rag_remove_document", {"document_id": "x"}),
    ])
    def test_tool_outputs_are_valid_json(self, tool_name, args):
        """All tool outputs should be parseable JSON."""
        rag_store = MagicMock()
        rag_store.search.return_value = []
        rag_store.list_documents.return_value = []
        rag_store.count.return_value = 0
        rag_store.remove_document.return_value = 0

        result = execute_tool(tool_name, args, rag_store)

        # Should not raise
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_unicode_preserved_in_output(self):
        """Non-ASCII characters preserved (ensure_ascii=False)."""
        rag_store = MagicMock()
        rag_store.search.return_value = [
            MagicMock(text="Cześć! Jak się masz?", score=0.9,
                      source="doc.md", document_id="d1"),
        ]

        result = execute_tool("rag_search", {"query": "test"}, rag_store)

        # Should contain Polish chars literally (not \uXXXX)
        assert "Cześć" in result or "Czes" in result