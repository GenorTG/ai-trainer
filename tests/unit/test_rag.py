"""Tests for inference_server/rag.py."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestSearchResult:
    def test_dataclass_fields(self):
        from inference_server.rag import SearchResult
        sr = SearchResult(text="hello", score=0.9, source="doc.txt", document_id="abc123")
        assert sr.text == "hello"
        assert sr.score == 0.9
        assert sr.source == "doc.txt"
        assert sr.document_id == "abc123"


@pytest.mark.unit
class TestRAGStore:
    def test_init_defaults(self):
        from inference_server.rag import RAGStore
        store = RAGStore()
        assert store.store_path == "rag_data/store"
        assert store._client is None
        assert store._collection is None

    def test_init_custom_path(self):
        from inference_server.rag import RAGStore
        store = RAGStore(store_path="/custom/path")
        assert store.store_path == "/custom/path"

    def test_get_client_caches(self, mocker):
        from inference_server.rag import RAGStore
        store = RAGStore(store_path="/tmp/test_rag")
        mock_chroma = mocker.MagicMock()
        mocker.patch.dict("sys.modules", {"chromadb": mock_chroma})
        c1 = store._get_client()
        c2 = store._get_client()
        assert c1 is c2

    def test_get_collection_caches(self, mocker):
        from inference_server.rag import RAGStore
        store = RAGStore(store_path="/tmp/test_rag")
        mock_chroma = mocker.MagicMock()
        mock_client = mock_chroma.PersistentClient.return_value
        mock_client.get_or_create_collection.return_value = mocker.MagicMock(name="collection")
        mocker.patch.dict("sys.modules", {"chromadb": mock_chroma})
        col1 = store._get_collection()
        col2 = store._get_collection()
        assert col1 is col2

    def test_get_embedder_caches(self, mocker):
        from inference_server.rag import RAGStore
        store = RAGStore()
        mock_st = mocker.MagicMock()
        mock_st.SentenceTransformer.return_value = mocker.MagicMock(name="embedder")
        mocker.patch.dict("sys.modules", {"sentence_transformers": mock_st})
        e1 = store._get_embedder()
        e2 = store._get_embedder()
        assert e1 is e2

    def test_add_document(self, mocker):
        from inference_server.rag import RAGStore
        store = RAGStore()
        mock_collection = mocker.MagicMock()
        mock_embedder = mocker.MagicMock()
        mock_embedder.encode.return_value.tolist.return_value = [[0.1] * 384, [0.2] * 384]
        mocker.patch.object(store, "_get_collection", return_value=mock_collection)
        mocker.patch.object(store, "_get_embedder", return_value=mock_embedder)

        chunks = [
            {"id": "c1", "text": "chunk one", "source": "a.txt"},
            {"id": "c2", "text": "chunk two", "source": "a.txt"},
        ]
        result = store.add_document("doc1", chunks, "all-MiniLM-L6-v2")
        assert result == 2
        mock_collection.upsert.assert_called_once()

    def test_search(self, mocker):
        from inference_server.rag import RAGStore
        store = RAGStore()
        mock_collection = mocker.MagicMock()
        mock_embedder = mocker.MagicMock()
        mock_embedder.encode.return_value.tolist.return_value = [[0.1] * 384]
        mock_collection.query.return_value = {
            "ids": [["c1"]],
            "documents": [["chunk text"]],
            "metadatas": [[{"source": "a.txt", "document_id": "doc1"}]],
            "distances": [[0.1]],
        }
        mocker.patch.object(store, "_get_collection", return_value=mock_collection)
        mocker.patch.object(store, "_get_embedder", return_value=mock_embedder)

        results = store.search("query", top_k=1)
        assert len(results) == 1
        assert results[0].text == "chunk text"
        assert results[0].score == pytest.approx(0.9)

    def test_search_empty_results(self, mocker):
        from inference_server.rag import RAGStore
        store = RAGStore()
        mock_collection = mocker.MagicMock()
        mock_embedder = mocker.MagicMock()
        mock_embedder.encode.return_value.tolist.return_value = [[0.1] * 384]
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mocker.patch.object(store, "_get_collection", return_value=mock_collection)
        mocker.patch.object(store, "_get_embedder", return_value=mock_embedder)

        results = store.search("no results")
        assert results == []

    def test_remove_document(self, mocker):
        from inference_server.rag import RAGStore
        store = RAGStore()
        mock_collection = mocker.MagicMock()
        mock_collection.get.return_value = {"ids": ["c1", "c2"]}
        mocker.patch.object(store, "_get_collection", return_value=mock_collection)

        removed = store.remove_document("doc1")
        assert removed == 2
        mock_collection.delete.assert_called_once_with(ids=["c1", "c2"])

    def test_remove_document_not_found(self, mocker):
        from inference_server.rag import RAGStore
        store = RAGStore()
        mock_collection = mocker.MagicMock()
        mock_collection.get.return_value = {"ids": []}
        mocker.patch.object(store, "_get_collection", return_value=mock_collection)

        removed = store.remove_document("nonexistent")
        assert removed == 0

    def test_list_documents(self, mocker):
        from inference_server.rag import RAGStore
        store = RAGStore()
        mock_collection = mocker.MagicMock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"document_id": "d1", "source": "a.txt"},
                {"document_id": "d1", "source": "b.txt"},
                {"document_id": "d2", "source": "c.txt"},
            ]
        }
        mocker.patch.object(store, "_get_collection", return_value=mock_collection)

        docs = store.list_documents()
        assert len(docs) == 2
        d1 = next(d for d in docs if d["document_id"] == "d1")
        assert d1["chunk_count"] == 2
        assert set(d1["sources"]) == {"a.txt", "b.txt"}

    def test_list_documents_empty(self, mocker):
        from inference_server.rag import RAGStore
        store = RAGStore()
        mock_collection = mocker.MagicMock()
        mock_collection.get.return_value = {"metadatas": []}
        mocker.patch.object(store, "_get_collection", return_value=mock_collection)

        assert store.list_documents() == []

    def test_count(self, mocker):
        from inference_server.rag import RAGStore
        store = RAGStore()
        mock_collection = mocker.MagicMock()
        mock_collection.count.return_value = 42
        mocker.patch.object(store, "_get_collection", return_value=mock_collection)

        assert store.count() == 42

    def test_clear(self, mocker):
        from inference_server.rag import RAGStore
        store = RAGStore()
        mock_client = mocker.MagicMock()
        store._client = mock_client
        store._collection = mocker.MagicMock()

        store.clear()
        mock_client.delete_collection.assert_called_once_with("documents")
        assert store._collection is None


@pytest.mark.unit
class TestDocumentIngestor:
    def test_init_defaults(self, mocker):
        from inference_server.rag import DocumentIngestor, RAGStore
        mock_store = mocker.MagicMock(spec=RAGStore)
        ingestor = DocumentIngestor(mock_store)
        assert ingestor.chunk_size == 512
        assert ingestor.overlap == 50

    def test_init_custom(self, mocker):
        from inference_server.rag import DocumentIngestor, RAGStore
        mock_store = mocker.MagicMock(spec=RAGStore)
        ingestor = DocumentIngestor(mock_store, chunk_size=256, overlap=30)
        assert ingestor.chunk_size == 256
        assert ingestor.overlap == 30

    def test_ingest_file(self, mocker, tmp_dir):
        from inference_server.rag import DocumentIngestor
        mock_store = mocker.MagicMock()
        mock_store.add_document.return_value = 1
        ingestor = DocumentIngestor(mock_store, chunk_size=512, overlap=50)
        p = tmp_dir / "test.txt"
        p.write_text("Hello world this is test content for ingestion.")
        result = ingestor.ingest_file(str(p))
        assert result["chunks"] >= 1
        mock_store.add_document.assert_called_once()

    def test_ingest_file_empty(self, mocker, tmp_dir):
        from inference_server.rag import DocumentIngestor
        mock_store = mocker.MagicMock()
        ingestor = DocumentIngestor(mock_store)
        p = tmp_dir / "empty.txt"
        p.write_text("")
        result = ingestor.ingest_file(str(p))
        assert result["chunks"] == 0
        assert "error" in result

    def test_ingest_directory(self, mocker, tmp_dir):
        from inference_server.rag import DocumentIngestor
        mock_store = mocker.MagicMock()
        mock_store.add_document.return_value = 2
        mock_store.count.return_value = 4
        ingestor = DocumentIngestor(mock_store)

        for i in range(3):
            (tmp_dir / f"file{i}.txt").write_text(f"Content for file {i}")
        (tmp_dir / "binary.bin").write_bytes(b"\x00\x01")

        result = ingestor.ingest_directory(str(tmp_dir))
        assert result["files_ingested"] == 3
        assert result["chunks_added"] >= 3

    def test_extract_text_txt(self, mocker, tmp_dir):
        from inference_server.rag import DocumentIngestor
        ingestor = DocumentIngestor(mocker.MagicMock())
        p = tmp_dir / "readme.txt"
        p.write_text("Hello")
        assert ingestor._extract_text(p) == "Hello"

    def test_extract_text_python(self, mocker, tmp_dir):
        from inference_server.rag import DocumentIngestor
        ingestor = DocumentIngestor(mocker.MagicMock())
        p = tmp_dir / "script.py"
        p.write_text("print('hi')")
        assert "print" in ingestor._extract_text(p)

    def test_chunk_text(self, mocker):
        from inference_server.rag import DocumentIngestor
        ingestor = DocumentIngestor(mocker.MagicMock(), chunk_size=3, overlap=1)
        words = ["a", "b", "c", "d", "e", "f", "g"]
        chunks = ingestor._chunk_text(" ".join(words))
        # chunk_size=3, overlap=1: first chunk = a b c, next starts at index 2
        assert len(chunks) >= 3
        assert "a b c" in chunks[0]

    def test_extract_pdf_fallback(self, mocker, tmp_dir):
        from inference_server.rag import DocumentIngestor
        ingestor = DocumentIngestor(mocker.MagicMock())
        p = tmp_dir / "doc.pdf"
        p.write_bytes(b"fake pdf")
        mocker.patch("subprocess.run", side_effect=FileNotFoundError)
        mocker.patch.dict("sys.modules", {"PyPDF2": None}, clear=False)
        result = ingestor._extract_pdf(p)
        assert isinstance(result, str)

    def test_extract_docx_fallback(self, mocker, tmp_dir):
        from inference_server.rag import DocumentIngestor
        ingestor = DocumentIngestor(mocker.MagicMock())
        p = tmp_dir / "doc.docx"
        p.write_bytes(b"fake docx")
        mocker.patch.dict("sys.modules", {"docx": None}, clear=False)
        result = ingestor._extract_docx(p)
        assert "python-docx" in result.lower() or "docx" in result.lower()
