"""Tests for finetune_studio.rag package.

Covers: store.py, ingest.py, manager.py, query.py.
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

# ══════════════════════════════════════════════════════════════
# ingest.py tests
# ══════════════════════════════════════════════════════════════

class TestChunkText:
    """Tests for chunk_text function."""

    def test_basic_chunking(self):
        """Text is split into chunks."""
        from finetune_studio.rag.ingest import chunk_text
        text = "word " * 100
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks) > 0
        assert all(c.text for c in chunks)

    def test_empty_text(self):
        """Empty text returns no chunks."""
        from finetune_studio.rag.ingest import chunk_text
        chunks = chunk_text("", chunk_size=10)
        assert chunks == []

    def test_whitespace_only(self):
        """Whitespace-only text returns no chunks."""
        from finetune_studio.rag.ingest import chunk_text
        chunks = chunk_text("   \n  ", chunk_size=10)
        assert chunks == []

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size gives one chunk."""
        from finetune_studio.rag.ingest import chunk_text
        chunks = chunk_text("hello world", chunk_size=100)
        assert len(chunks) == 1

    def test_chunk_indices(self):
        """Chunks have sequential indices."""
        from finetune_studio.rag.ingest import chunk_text
        text = "word " * 200
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_overlap(self):
        """Consecutive chunks share words via overlap."""
        from finetune_studio.rag.ingest import chunk_text
        text = " ".join(f"word{i}" for i in range(100))
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        if len(chunks) >= 2:
            words1 = set(chunks[0].text.split())
            words2 = set(chunks[1].text.split())
            assert len(words1 & words2) > 0

    def test_metadata_propagation(self):
        """Metadata is passed to all chunks."""
        from finetune_studio.rag.ingest import chunk_text
        meta = {"source": "test.txt"}
        chunks = chunk_text("hello world testing data", chunk_size=3, overlap=1, metadata=meta)
        for c in chunks:
            assert c.metadata == meta

    def test_chunk_ids_assigned_later(self):
        """Chunks start with empty id (assigned by ingest)."""
        from finetune_studio.rag.ingest import chunk_text
        chunks = chunk_text("hello world", chunk_size=100)
        assert chunks[0].id == ""


class TestExtractText:
    """Tests for extract_text function."""

    def test_txt_file(self, tmp_dir):
        """Extracts text from .txt file."""
        from finetune_studio.rag.ingest import extract_text
        p = tmp_dir / "test.txt"
        p.write_text("Hello, world!")
        assert extract_text(str(p)) == "Hello, world!"

    def test_md_file(self, tmp_dir):
        """Extracts text from .md file."""
        from finetune_studio.rag.ingest import extract_text
        p = tmp_dir / "test.md"
        p.write_text("# Title\n\nContent here.")
        assert "Title" in extract_text(str(p))

    def test_python_file(self, tmp_dir):
        """Extracts text from .py file."""
        from finetune_studio.rag.ingest import extract_text
        p = tmp_dir / "test.py"
        p.write_text("def hello():\n    pass")
        assert "hello" in extract_text(str(p))

    def test_unknown_extension(self, tmp_dir):
        """Unknown extension tries reading as text."""
        from finetune_studio.rag.ingest import extract_text
        p = tmp_dir / "test.xyz"
        p.write_text("some content")
        assert extract_text(str(p)) == "some content"


class TestDocumentDataclass:
    """Tests for Document dataclass."""

    def test_document_defaults(self):
        """Document has empty defaults."""
        from finetune_studio.rag.ingest import Document
        d = Document()
        assert d.id == ""
        assert d.chunks == []
        assert d.chunk_count == 0

    def test_document_with_data(self):
        """Document stores provided data."""
        from finetune_studio.rag.ingest import Chunk, Document
        c = Chunk(text="hello", chunk_index=0)
        d = Document(id="abc", filename="test.txt", chunks=[c], chunk_count=1)
        assert d.id == "abc"
        assert len(d.chunks) == 1


class TestIngestFile:
    """Tests for ingest_file function."""

    def test_ingest_txt_file(self, tmp_dir):
        """Ingests a text file and returns Document with chunks."""
        from finetune_studio.rag.ingest import ingest_file
        p = tmp_dir / "test.txt"
        p.write_text("This is a test document. " * 50)
        doc = ingest_file(str(p), chunk_size=20, overlap=5)
        assert doc.filename == "test.txt"
        assert doc.chunk_count > 0
        assert len(doc.chunks) == doc.chunk_count
        assert doc.id

    def test_ingest_file_ids_assigned(self, tmp_dir):
        """Chunks get document_id and individual ids."""
        from finetune_studio.rag.ingest import ingest_file
        p = tmp_dir / "test.txt"
        p.write_text("Word " * 100)
        doc = ingest_file(str(p), chunk_size=20, overlap=5)
        for i, chunk in enumerate(doc.chunks):
            assert chunk.document_id == doc.id
            assert chunk.id == f"{doc.id}_{i}"


class TestIngestDirectory:
    """Tests for ingest_directory function."""

    def test_ingest_directory(self, tmp_dir):
        """Ingests all supported files in directory."""
        from finetune_studio.rag.ingest import ingest_directory
        d = tmp_dir / "docs"
        d.mkdir()
        (d / "a.txt").write_text("Document A. " * 30)
        (d / "b.md").write_text("Document B. " * 30)
        (d / "c.xyz").write_text("Ignored file.")
        docs = ingest_directory(str(d), chunk_size=20, overlap=5, extensions=[".txt", ".md"])
        assert len(docs) == 2

    def test_ingest_directory_empty(self, tmp_dir):
        """Empty directory returns no documents."""
        from finetune_studio.rag.ingest import ingest_directory
        d = tmp_dir / "empty"
        d.mkdir()
        docs = ingest_directory(str(d))
        assert docs == []


# ══════════════════════════════════════════════════════════════
# store.py tests
# ══════════════════════════════════════════════════════════════

class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_defaults(self):
        """SearchResult has defaults."""
        from finetune_studio.rag.store import SearchResult
        sr = SearchResult()
        assert sr.chunk_id == ""
        assert sr.score == 0.0
        assert sr.metadata == {}

    def test_search_result_with_data(self):
        """SearchResult stores provided data."""
        from finetune_studio.rag.store import SearchResult
        sr = SearchResult(chunk_id="c1", text="hello", score=0.95, source="test.txt")
        assert sr.chunk_id == "c1"
        assert sr.score == 0.95


class TestVectorStore:
    """Tests for VectorStore class (mocked ChromaDB)."""

    def _mock_chromadb(self):
        """Create a mock chromadb module."""
        mock_chromadb = MagicMock()
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection
        return mock_chromadb, mock_client, mock_collection

    def test_init_creates_path(self):
        """VectorStore initializes with store path."""
        from finetune_studio.rag.store import VectorStore
        vs = VectorStore(store_path="/tmp/test_store")
        assert vs.store_path == "/tmp/test_store"

    def test_get_client_creates_once(self):
        """_get_client creates client once and reuses."""
        from finetune_studio.rag.store import VectorStore
        mock_chromadb, mock_client, _ = self._mock_chromadb()
        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            vs = VectorStore()
            c1 = vs._get_client()
            c2 = vs._get_client()
            assert c1 is c2
            mock_chromadb.PersistentClient.assert_called_once()

    def test_get_collection(self):
        """_get_collection creates/get collection."""
        from finetune_studio.rag.store import VectorStore
        mock_chromadb, mock_client, mock_collection = self._mock_chromadb()
        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            vs = VectorStore()
            col = vs._get_collection()
            assert col is mock_collection
            mock_client.get_or_create_collection.assert_called_once()

    def test_add_chunks(self):
        """add_chunks embeds and stores chunks."""
        from finetune_studio.rag.ingest import Chunk
        from finetune_studio.rag.store import VectorStore
        mock_chromadb, mock_client, mock_collection = self._mock_chromadb()
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value.tolist.return_value = [[0.1] * 384]
        with patch.dict(sys.modules, {"chromadb": mock_chromadb, "sentence_transformers": MagicMock(SentenceTransformer=MagicMock(return_value=mock_embedder))}):
            vs = VectorStore()
            # Force lazy init
            vs._collection = mock_collection
            vs._embedder = mock_embedder
            chunks = [Chunk(id="c1", text="hello", document_id="d1", chunk_index=0)]
            count = vs.add_chunks(chunks)
            assert count == 1
            mock_collection.upsert.assert_called_once()

    def test_search_returns_results(self):
        """search returns SearchResult list."""
        from finetune_studio.rag.store import VectorStore
        mock_chromadb, mock_client, mock_collection = self._mock_chromadb()
        mock_collection.query.return_value = {
            "ids": [["c1"]],
            "documents": [["hello world"]],
            "metadatas": [[{"document_id": "d1", "source": "test.txt"}]],
            "distances": [[0.1]],
        }
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value.tolist.return_value = [[0.1] * 384]
        with patch.dict(sys.modules, {"chromadb": mock_chromadb, "sentence_transformers": MagicMock(SentenceTransformer=MagicMock(return_value=mock_embedder))}):
            vs = VectorStore()
            vs._collection = mock_collection
            vs._embedder = mock_embedder
            results = vs.search("hello", top_k=1)
            assert len(results) == 1
            assert results[0].text == "hello world"
            assert results[0].score == pytest.approx(0.9, abs=0.01)

    def test_remove_document(self):
        """remove_document deletes chunks by document_id."""
        from finetune_studio.rag.store import VectorStore
        mock_chromadb, mock_client, mock_collection = self._mock_chromadb()
        mock_collection.get.return_value = {"ids": ["c1", "c2"]}
        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            vs = VectorStore()
            vs._collection = mock_collection
            count = vs.remove_document("d1")
            assert count == 2
            mock_collection.delete.assert_called_once_with(ids=["c1", "c2"])

    def test_list_documents(self):
        """list_documents groups by document_id."""
        from finetune_studio.rag.store import VectorStore
        mock_chromadb, mock_client, mock_collection = self._mock_chromadb()
        mock_collection.get.return_value = {
            "metadatas": [
                {"document_id": "d1", "source": "a.txt"},
                {"document_id": "d1", "source": "a.txt"},
                {"document_id": "d2", "source": "b.txt"},
            ]
        }
        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            vs = VectorStore()
            vs._collection = mock_collection
            docs = vs.list_documents()
            assert len(docs) == 2
            assert docs[0]["chunk_count"] == 2

    def test_count(self):
        """count returns total chunks."""
        from finetune_studio.rag.store import VectorStore
        mock_chromadb, mock_client, mock_collection = self._mock_chromadb()
        mock_collection.count.return_value = 42
        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            vs = VectorStore()
            vs._collection = mock_collection
            assert vs.count() == 42

    def test_clear(self):
        """clear deletes collection."""
        from finetune_studio.rag.store import VectorStore
        mock_chromadb, mock_client, mock_collection = self._mock_chromadb()
        with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
            vs = VectorStore()
            vs._client = mock_client
            vs.clear()
            mock_client.delete_collection.assert_called_once_with("documents")


# ══════════════════════════════════════════════════════════════
# query.py tests
# ══════════════════════════════════════════════════════════════

class TestRAGConfig:
    """Tests for RAGConfig dataclass."""

    def test_defaults(self):
        """RAGConfig has sensible defaults."""
        from finetune_studio.rag.query import RAGConfig
        c = RAGConfig()
        assert c.top_k == 5
        assert c.min_score == 0.3
        assert c.max_context_length == 2000

    def test_custom(self):
        """RAGConfig accepts custom values."""
        from finetune_studio.rag.query import RAGConfig
        c = RAGConfig(top_k=10, min_score=0.5)
        assert c.top_k == 10
        assert c.min_score == 0.5


class TestRAGQuery:
    """Tests for RAGQuery class."""

    def _make_search_result(self, text, score, source="test.txt", doc_id="d1"):
        from finetune_studio.rag.store import SearchResult
        return SearchResult(text=text, score=score, source=source, document_id=doc_id,
                           metadata={"source": source})

    def test_retrieve_filters_by_min_score(self):
        """retrieve filters results below min_score."""
        from finetune_studio.rag.query import RAGConfig, RAGQuery
        mock_store = MagicMock()
        r1 = self._make_search_result("relevant", 0.8)
        r2 = self._make_search_result("irrelevant", 0.1)
        mock_store.search.return_value = [r1, r2]
        config = RAGConfig(min_score=0.3)
        rag = RAGQuery(mock_store, config)
        results = rag.retrieve("query")
        assert len(results) == 1
        assert results[0].text == "relevant"

    def test_retrieve_top_k(self):
        """retrieve respects top_k parameter."""
        from finetune_studio.rag.query import RAGQuery
        mock_store = MagicMock()
        mock_store.search.return_value = [self._make_search_result(f"r{i}", 0.9) for i in range(10)]
        rag = RAGQuery(mock_store)
        rag.retrieve("query", top_k=3)
        mock_store.search.assert_called_once_with("query", top_k=3)

    def test_build_context_empty(self):
        """build_context returns empty string when no results."""
        from finetune_studio.rag.query import RAGQuery
        mock_store = MagicMock()
        mock_store.search.return_value = []
        rag = RAGQuery(mock_store)
        ctx = rag.build_context("query")
        assert ctx == ""

    def test_build_context_with_results(self):
        """build_context formats results into context."""
        from finetune_studio.rag.query import RAGQuery
        mock_store = MagicMock()
        r1 = self._make_search_result("First chunk", 0.9, source="a.txt")
        r2 = self._make_search_result("Second chunk", 0.8, source="b.txt")
        mock_store.search.return_value = [r1, r2]
        rag = RAGQuery(mock_store)
        ctx = rag.build_context("query")
        assert "First chunk" in ctx
        assert "Second chunk" in ctx
        assert "---" in ctx

    def test_build_context_max_length(self):
        """build_context respects max_context_length."""
        from finetune_studio.rag.query import RAGConfig, RAGQuery
        mock_store = MagicMock()
        r1 = self._make_search_result("A" * 3000, 0.9)
        mock_store.search.return_value = [r1]
        config = RAGConfig(max_context_length=100)
        rag = RAGQuery(mock_store, config)
        ctx = rag.build_context("query")
        assert len(ctx) <= 100 + 200

    def test_augment_prompt_with_context(self):
        """augment_prompt includes context in messages."""
        from finetune_studio.rag.query import RAGQuery
        mock_store = MagicMock()
        r1 = self._make_search_result("Relevant info", 0.9)
        mock_store.search.return_value = [r1]
        rag = RAGQuery(mock_store)
        messages = rag.augment_prompt("What is AI?", system_prompt="Be helpful")
        assert len(messages) >= 2
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What is AI?"

    def test_augment_prompt_no_context(self):
        """augment_prompt still returns user message when no context."""
        from finetune_studio.rag.query import RAGQuery
        mock_store = MagicMock()
        mock_store.search.return_value = []
        rag = RAGQuery(mock_store)
        messages = rag.augment_prompt("What is AI?")
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "What is AI?"

    def test_query_full_rag(self):
        """query calls inference engine and returns structured result."""
        from finetune_studio.rag.query import RAGQuery
        mock_store = MagicMock()
        r1 = self._make_search_result("RAG context here", 0.9)
        mock_store.search.return_value = [r1]
        mock_engine = MagicMock()
        mock_engine.generate.return_value = {"response": "AI is artificial intelligence"}
        rag = RAGQuery(mock_store)
        result = rag.query(mock_engine, "What is AI?")
        assert "response" in result
        assert "sources" in result
        assert "context_used" in result
        assert result["chunks_retrieved"] == 1
        mock_engine.generate.assert_called_once()


# ══════════════════════════════════════════════════════════════
# manager.py tests
# ══════════════════════════════════════════════════════════════

class TestRAGManager:
    """Tests for RAGManager class."""

    def test_init(self):
        """RAGManager initializes with store path."""
        from finetune_studio.rag.manager import RAGManager
        with patch("finetune_studio.rag.manager.VectorStore"):
            mgr = RAGManager(store_path="/tmp/test")
            assert mgr.store is not None

    def test_ingest_file(self, tmp_dir):
        """ingest_file processes a file and adds to store."""
        from finetune_studio.rag.manager import RAGManager
        p = tmp_dir / "test.txt"
        p.write_text("Hello world. " * 30)
        with patch("finetune_studio.rag.manager.VectorStore") as MockStore:
            mock_store = MagicMock()
            mock_store.add_chunks.return_value = 3
            mock_store.count.return_value = 3
            MockStore.return_value = mock_store
            mgr = RAGManager()
            result = mgr.ingest_file(str(p))
            assert result["chunks_added"] == 3
            assert result["filename"] == "test.txt"

    def test_ingest_directory(self, tmp_dir):
        """ingest_directory processes multiple files."""
        from finetune_studio.rag.manager import RAGManager
        d = tmp_dir / "docs"
        d.mkdir()
        (d / "a.txt").write_text("Doc A. " * 30)
        (d / "b.txt").write_text("Doc B. " * 30)
        with patch("finetune_studio.rag.manager.VectorStore") as MockStore:
            mock_store = MagicMock()
            mock_store.add_chunks.return_value = 2
            mock_store.count.return_value = 4
            MockStore.return_value = mock_store
            mgr = RAGManager()
            result = mgr.ingest_directory(str(d))
            assert result["documents_ingested"] == 2

    def test_remove_document(self):
        """remove_document calls store.remove_document."""
        from finetune_studio.rag.manager import RAGManager
        with patch("finetune_studio.rag.manager.VectorStore") as MockStore:
            mock_store = MagicMock()
            mock_store.remove_document.return_value = 5
            mock_store.count.return_value = 10
            MockStore.return_value = mock_store
            mgr = RAGManager()
            result = mgr.remove_document("d1")
            assert result["chunks_removed"] == 5
            mock_store.remove_document.assert_called_once_with("d1")

    def test_list_documents(self):
        """list_documents returns store documents."""
        from finetune_studio.rag.manager import RAGManager
        with patch("finetune_studio.rag.manager.VectorStore") as MockStore:
            mock_store = MagicMock()
            mock_store.list_documents.return_value = [{"document_id": "d1"}]
            MockStore.return_value = mock_store
            mgr = RAGManager()
            docs = mgr.list_documents()
            assert len(docs) == 1

    def test_stats(self):
        """stats returns chunk and document counts."""
        from finetune_studio.rag.manager import RAGManager
        with patch("finetune_studio.rag.manager.VectorStore") as MockStore:
            mock_store = MagicMock()
            mock_store.count.return_value = 42
            mock_store.list_documents.return_value = [{"document_id": "d1"}, {"document_id": "d2"}]
            MockStore.return_value = mock_store
            mgr = RAGManager()
            stats = mgr.stats()
            assert stats["total_chunks"] == 42
            assert stats["total_documents"] == 2

    def test_clear(self):
        """clear calls store.clear."""
        from finetune_studio.rag.manager import RAGManager
        with patch("finetune_studio.rag.manager.VectorStore") as MockStore:
            mock_store = MagicMock()
            MockStore.return_value = mock_store
            mgr = RAGManager()
            mgr.clear()
            mock_store.clear.assert_called_once()
